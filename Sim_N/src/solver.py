"""
src/solver.py


Simulation pipeline stages
--------------------------
1. Mesh + function spaces  (meshing.py)
       Layered tetrahedral spherical-cap mesh.
       5-component CG1 Q-tensor function space.

2. Initial conditions  (initial_conditions.py)
       Populate U_current, U_previous, U_prev_prev with the chosen IC type.
       BDF2 requires two history states; both are initialised from the same IC.

3. Boundary conditions  (boundary_conditions.py)
       Hard Dirichlet BCs at the dome rim (azimuthal director).
       Soft penalty terms (anchoring) are part of the weak form below.

4. Weak form assembly  (GinzburgLandauSolver)
       Elastic:       L1/2·|∇Q|² + L2/2·|div Q|² + L3/2·Q_{ij,k}Q_{ik,j}
       Dome anchor:   C_surface/2·(ν^T Q ν + S/3)²       [planar degenerate]
       Base anchor:   C_polyamide/2·|Q − Q_s|²                [Rapini-Papoular]
                    + C_surface/2·(ẑ^T Q ẑ + S/3)²           [planar degenerate]
       Transient:     γ⁻¹·dQ/dt  (BDF1 first step, BDF2 thereafter)

5. Time stepping  (GinzburgLandauSolver.run_time_stepping)
       BDF1 for step 1 (single-step start-up).
       BDF2 from step 2 onward (second-order accurate).
       Adaptive dt: halved on SNES failure, up to max_retries attempts per step.

6. Output  (output_handler.py)
       Director extracted via eigendecomposition and written to XDMF every
       output_every_n_steps steps.  Energy components logged to CSV.

Functions
---------
_make_Q
    Build the 3×3 symmetric traceless Q-tensor from its 5-component vector.
run_simulation
    Orchestrate stages 1–6 for a complete simulation run.

Classes
-------
GinzburgLandauSolver
    Assembles and solves the Q-tensor Ginzburg-Landau PDE system.
"""

import numpy as np
import os
import csv
import traceback
from progiter import ProgIter

from mpi4py import MPI
from dolfinx import fem as _fem
from petsc4py import PETSc
import ufl
from ufl import (Measure, as_vector, as_tensor, outer, Identity,
                 dot, derivative, TestFunction, TrialFunction,
                 inner, grad, div, sqrt, SpatialCoordinate)


# ---------------------------------------------------------------------------
# Helper: 5-component vector → 3×3 Q-tensor (UFL)
# ---------------------------------------------------------------------------

def _make_Q(q):
    """
    Build the symmetric traceless 3×3 Q-tensor from its 5-component UFL vector.

    Parameters
    ----------
    q : ufl.Argument or ufl.Function (5-component)
        Q-tensor components [q0, q1, q2, q3, q4].

    Returns
    -------
    ufl.tensors.ComponentTensor, shape (3, 3)
        Symmetric traceless tensor:
            Q = [[ q0,   q2,   q3  ],
                 [ q2,   q1,   q4  ],
                 [ q3,   q4,  -q0-q1]]
    """
    return as_tensor([[q[0],        q[2],        q[3]       ],
                      [q[2],        q[1],        q[4]       ],
                      [q[3],        q[4],       -q[0]-q[1]  ]])


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class GinzburgLandauSolver:
    """
    Assembles and solves the Q-tensor Ginzburg-Landau dynamics.

    Time-stepping uses BDF1 for the first step (self-starting) and BDF2 for
    all subsequent steps (second-order accurate).  Each step is solved with
    PETSc SNES (Newton line-search) using a direct MUMPS LU factorisation.

    The total energy functional is:

        F = ∫_Ω  L1/2·|∇Q|² + L2/2·|div Q|² + L3/2·Q_{ij,k}Q_{ik,j}  dV
          + ∫_dome  C_surface/2·(ν^T Q ν + S/3)²                        dS
          + ∫_base  C_polyamide/2·|Q − Q_s_base|²                           dS
          + ∫_base  C_surface/2·(ẑ^T Q ẑ + S/3)²                           dS

    where L1, L2, L3 are derived from the Frank elastic constants K1, K2, K3
    via the standard LdG–Frank mapping at order parameter S.

    Parameters
    ----------
    params : SimulationParameters
        All physical and numerical parameters.
    spaces : FunctionSpaces
        Q-tensor function space container.
    mesh_handler : MeshHandler
        Mesh and meshtag container.
    bcs_list : list[fem.DirichletBC]
        Hard Dirichlet BCs (dome rim).
    output_handler : OutputHandler
        XDMF output writer.
    """

    def __init__(self, params, spaces, mesh_handler, bcs_list, output_handler):
        if MPI.COMM_WORLD.rank == 0:
            print("GinzburgLandauSolver: Initializing...")
        self.params = params
        self.spaces = spaces
        self.mesh_handler = mesh_handler
        self.bcs_list = bcs_list
        self.output_handler = output_handler
        self.energy_history = []

        # Time-stepping parameters
        self.dt = _fem.Constant(spaces.mesh, PETSc.ScalarType(params.dt))
        self.inv_gamma_dt = _fem.Constant(
            spaces.mesh,
            PETSc.ScalarType(1.0 / (params.gamma_viscosity * params.dt))
        )

        # BDF coefficients (start with BDF1, switch to BDF2 after first step)
        # BDF1: a0=1, a1=-1, a2=0  →  dP/dt ≈ (P^{n+1} - P^n) / dt
        # BDF2: a0=3/2, a1=-2, a2=1/2  →  dP/dt ≈ (3/2*P^{n+1} - 2*P^n + 1/2*P^{n-1}) / dt
        self.bdf_a0 = _fem.Constant(spaces.mesh, PETSc.ScalarType(1.0))
        self.bdf_a1 = _fem.Constant(spaces.mesh, PETSc.ScalarType(-1.0))
        self.bdf_a2 = _fem.Constant(spaces.mesh, PETSc.ScalarType(0.0))

        # Measures
        self.dx = Measure("dx", domain=spaces.mesh, subdomain_data=mesh_handler.cell_tags)
        self.ds = Measure("ds", domain=spaces.mesh, subdomain_data=mesh_handler.boundary_tags)

        # Dome normal: exact outward normal for a sphere of radius R_sphere
        # centered at (0, 0, z_center).  n = (x, y, z - z_center) / R_sphere.
        R_base = params.base_radius
        theta = params.contact_angle_rad
        R_sphere = R_base / np.sin(theta)
        z_center = -R_sphere * np.cos(theta)

        x = SpatialCoordinate(spaces.mesh)

        n_sph_unnorm = as_vector([x[0], x[1], x[2] - z_center])
        n_sph_mag = sqrt(dot(n_sph_unnorm, n_sph_unnorm) + 1e-24)
        self.n_dome = n_sph_unnorm / n_sph_mag

        # Base normal: outward normal for z=0 plane is -z, but ν⊗ν is the same for ±z
        self.n_base = as_vector([0.0, 0.0, 1.0])

        # Base rubbing direction: normalised in-plane (z=0) vector
        d_xy = params.D_rub / (np.linalg.norm(params.D_rub) + 1e-16)
        self.d_rub = as_vector([float(d_xy[0]), float(d_xy[1]), 0.0])

        # Preferred Q-tensor for base: Q_s = S*(d̂⊗d̂ - I/3)
        self.Q_s_base = params.S * (outer(self.d_rub, self.d_rub) - Identity(3) / 3.0)

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        if MPI.COMM_WORLD.rank == 0:
            if not os.path.exists(params.output_dir):
                os.makedirs(params.output_dir)
            csv_path = os.path.join(params.output_dir, "dynamic_log.csv")
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Time (s)', 'Iteration', 'Total Energy (J)',
                'Elastic (J)', 'Dome Anchoring (J)', 'Base Anchoring (J)'
            ])

    def get_steady_state_residual_forms(self, U, v):
        """
        Assemble the steady-state residual δF/δQ tested against v.

        Parameters
        ----------
        U : ufl.Argument or fem.Function (5-component)
            Current Q-tensor field (trial / solution).
        v : ufl.Argument (5-component)
            Test function.

        Returns
        -------
        ufl.Form
            Residual form R(U, v) summing elastic and surface anchoring contributions.
        """
        Q   = _make_Q(U)
        V   = _make_Q(v)
        p   = self.params

        ID_CAP    = self.mesh_handler.ID_CAP
        ID_DOME   = self.mesh_handler.ID_DOME
        ID_BOTTOM = self.mesh_handler.ID_BOTTOM

        # 1. Elastic: L1/2 |∇Q|² → δ = L1 * inner(grad Q, grad V)
        R  = p.L1 * inner(grad(Q), grad(V)) * self.dx(ID_CAP)

        # 2. Elastic: L2/2 |div Q|² → δ = L2 * inner(div Q, div V)
        R += p.L2 * inner(div(Q), div(V)) * self.dx(ID_CAP)

        # 3. Elastic: L3/2 Q_{ij,k}Q_{ik,j}
        # δ w.r.t. Q: L3 * dQ_swap_{ijk} * dV_{ijk}  where dQ_swap[i,j,k] = ∂Q_{ik}/∂x_j
        dQ = grad(Q)   # dQ[i,j,k] = ∂Q_{ij}/∂x_k
        dQ_swap = as_tensor([[[dQ[i, k, j] for k in range(3)]
                               for j in range(3)]
                              for i in range(3)])
        R += p.L3 * inner(dQ_swap, grad(V)) * self.dx(ID_CAP)

        # 5. Dome anchoring: planar degenerate - penalise normal component of Q
        # F = C/2 * (νᵀQν + S/3)²   →   δF = C * (νᵀQν + S/3) * (νᵀVν)
        nQn = inner(outer(self.n_dome, self.n_dome), Q)   # ν^T Q ν
        nVn = inner(outer(self.n_dome, self.n_dome), V)   # ν^T V ν
        R += p.C_surface * (nQn + p.S / 3.0) * nVn * self.ds(ID_DOME)

        # 6. Base anchoring: Rapini-Papoular toward d̂  →  δF = C * (Q - Q_s_base) : V
        dQ_base = Q - self.Q_s_base
        R += p.C_polyamide * inner(dQ_base, V) * self.ds(ID_BOTTOM)

        # 7. Base planar degenerate: same form as dome, normal = ẑ
        nQn_b = inner(outer(self.n_base, self.n_base), Q)
        nVn_b = inner(outer(self.n_base, self.n_base), V)
        R += p.C_surface * (nQn_b + p.S / 3.0) * nVn_b * self.ds(ID_BOTTOM)

        return R

    def build_dynamic_residual_form(self, U_current, U_previous, U_prev_prev, v):
        """
        Assemble the full time-dependent residual (transient + steady-state).

        The time derivative is approximated via BDF coefficients stored as
        fem.Constants so the form can be compiled once and updated each step:
            BDF1 (step 1):   dQ/dt ≈ (a0·Q^{n+1} + a1·Q^n) / dt
            BDF2 (step 2+):  dQ/dt ≈ (a0·Q^{n+1} + a1·Q^n + a2·Q^{n-1}) / dt

        Parameters
        ----------
        U_current : fem.Function
            Q-tensor at the new time level (n+1), the unknown.
        U_previous : fem.Function
            Q-tensor at the previous time level (n).
        U_prev_prev : fem.Function
            Q-tensor two steps back (n-1); used only for BDF2.
        v : ufl.Argument
            Test function.

        Returns
        -------
        ufl.Form
            Combined transient + steady-state residual form.
        """
        R_steady = self.get_steady_state_residual_forms(U_current, v)

        R_transient = self.inv_gamma_dt * inner(
            self.bdf_a0 * U_current + self.bdf_a1 * U_previous + self.bdf_a2 * U_prev_prev,
            v
        ) * self.dx(self.mesh_handler.ID_CAP)

        return R_transient + R_steady

    def run_time_stepping(self, U_current, U_previous, U_prev_prev):
        """
        Execute the BDF1→BDF2 time-stepping loop.

        On SNES failure the time step is halved and the step retried from the
        saved snapshot, up to max_retries times.  A "soft" acceptance criterion
        allows SNES to exit on stol/local-min reasons when the residual is already
        far below the initial driving residual (common in SI-unit simulations where
        assembled residuals are numerically tiny).

        Parameters
        ----------
        U_current : fem.Function
            Q-tensor at the current (new) time level - updated in place.
        U_previous : fem.Function
            Q-tensor at the previous time level - updated in place after each step.
        U_prev_prev : fem.Function
            Q-tensor two steps back - updated in place after each step.
        """
        from .snes_problem import SNESProblem

        comm = MPI.COMM_WORLD
        rank = comm.rank
        if rank == 0:
            print("\n" + "=" * 70)
            print(f" Starting Ginzburg-Landau Dynamic Simulation")
            print(f"  dt = {self.params.dt:.1e} s,  Num Steps = {self.params.num_steps}")
            print(f"  gamma_visc = {self.params.gamma_viscosity:.1e}")
            print(f"  Time-stepping: BDF1 (1st step) -> BDF2 (thereafter)")
            print("=" * 70 + "\n")

        # Setup Output
        p_out = self.output_handler.setup_output_files(self.spaces)
        self.output_handler.write_timestep(p_out, U_current, 0.0)

        # Setup SNES
        snes = PETSc.SNES().create(comm)
        snes.setType("newtonls")
        # SI units make assembled residuals ~1e-13 (tiny domain, tiny constants).
        # Use stol (step-size) as primary and rtol (relative residual decrease) as secondary.
        # atol is set after computing initial residual (problem-scale-aware).
        snes.setTolerances(atol=1e-30, rtol=1e-8, stol=1e-10,
                           max_it=self.params.newton_max_iter)
        PETSc.Options()["snes_divergence_tolerance"] = 1e30

        # Direct solver (robust for small-medium meshes):
        ksp = snes.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        PETSc.Options()["snes_linesearch_type"] = "bt"
        snes.setFromOptions()
        snes.setForceIteration(True)

        # Build residual and Jacobian forms once
        v = TestFunction(self.spaces.W)
        dU_trial = TrialFunction(self.spaces.W)

        R_dynamic_form = self.build_dynamic_residual_form(U_current, U_previous, U_prev_prev, v)
        J_dynamic_form = derivative(R_dynamic_form, U_current, dU_trial)

        # Create SNESProblem instance once
        problem = SNESProblem(R_dynamic_form, J_dynamic_form, U_current, self.bcs_list)

        # Set SNES function and Jacobian callbacks once
        snes.setFunction(problem.F, problem.b)
        snes.setJacobian(problem.J, J=problem.A, P=None)

        # Compute reference residual scale (steady-state residual of IC).
        # Used to accept "soft" SNES failures (max_it, local_min) when the
        # residual is already tiny relative to the initial driving force.
        problem.F(snes, U_current.x.petsc_vec, problem.b)
        r0_ref = problem.b.norm()
        if rank == 0:
            print(f"  Reference residual (IC): {r0_ref:.2e}")

        # Main Time Loop
        t = 0.0
        dt = self.params.dt
        last_E_total = np.nan
        max_retries = 5

        # Snapshot for retry on SNES failure
        U_snap = np.empty_like(U_current.x.array)
        U_prev_snap = np.empty_like(U_previous.x.array)
        U_pp_snap = np.empty_like(U_prev_prev.x.array)

        # Start with BDF1 coefficients
        self.bdf_a0.value = 1.0
        self.bdf_a1.value = -1.0
        self.bdf_a2.value = 0.0

        pbar = ProgIter(total=self.params.num_steps, desc="Time step",
                        verbose=1 if rank == 0 else 0, time_thresh=2.0)
        pbar.begin()

        for n_step in range(1, self.params.num_steps + 1):
            t = n_step * self.params.dt
            self.dt.value = dt
            self.inv_gamma_dt.value = 1.0 / (self.params.gamma_viscosity * dt)

            # Switch to BDF2 after first step
            if n_step == 2:
                self.bdf_a0.value = 1.5
                self.bdf_a1.value = -2.0
                self.bdf_a2.value = 0.5

            # Save snapshot for retry
            U_snap[:] = U_current.x.array
            U_prev_snap[:] = U_previous.x.array
            U_pp_snap[:] = U_prev_prev.x.array

            # SNES solve (with retry on failure)
            converged = False
            retry_dt = dt
            snes_status = ""
            for attempt in range(max_retries + 1):
                snes.solve(None, U_current.x.petsc_vec)
                U_current.x.scatter_forward()

                reason = snes.getConvergedReason()
                n_iter = snes.getIterationNumber()
                norm_residual = snes.getFunctionNorm()

                if reason > 0:
                    converged = True
                    snes_status = f"ok({n_iter})"
                    break

                # Accept "soft" failures when residual is already well below
                # the initial driving residual (solution is effectively found,
                # but stol/bt can't verify due to SI-unit noise floor).
                if reason in (-5, -8) and (norm_residual < r0_ref * 1e-6 or norm_residual < 1e-12):
                    converged = True
                    snes_status = f"ok~({n_iter})"
                    break

                # Failed - restore snapshot, halve dt, retry
                U_current.x.array[:] = U_snap
                U_previous.x.array[:] = U_prev_snap
                U_prev_prev.x.array[:] = U_pp_snap
                U_current.x.scatter_forward()
                U_previous.x.scatter_forward()
                U_prev_prev.x.scatter_forward()

                old_dt = retry_dt
                retry_dt = retry_dt * 0.5
                self.dt.value = retry_dt
                self.inv_gamma_dt.value = 1.0 / (self.params.gamma_viscosity * retry_dt)
                snes_status = f"r{attempt+1}({reason})"

            if not converged:
                if rank == 0:
                    print(f"\n! SNES did not converge at step {n_step} (t={t:.1e})"
                          f" after {max_retries} retries!"
                          f" Reason: {reason}, Residual: {norm_residual:.2e}")
                comm.Abort(1)

            # Restore nominal dt for next step
            step_dt = retry_dt  # actual dt used for this step
            dt = self.params.dt

            # Update history: prev_prev ← previous, previous ← current
            U_prev_prev.x.array[:] = U_previous.x.array
            U_previous.x.array[:] = U_current.x.array

            # Logging and Output (conditional block)
            if n_step % self.params.output_every_n_steps == 0 or n_step == self.params.num_steps:
                E_dict = self.compute_energy(U_current)
                E_total = E_dict['total']
                self.energy_history.append(E_total)
                last_E_total = E_total

                if rank == 0:
                    self.csv_writer.writerow([
                        f"{t:.6e}",
                        n_step,
                        f"{E_total:.6e}",
                        f"{E_dict['elastic']:.6e}",
                        f"{E_dict['surf_dome']:.6e}",
                        f"{E_dict['surf_base']:.6e}"
                    ])
                    self.csv_file.flush()

                self.output_handler.write_timestep(p_out, U_current, t)

            extra = {
                'E': f'{last_E_total:.3e}',
                'snes': snes_status,
                'res': f'{norm_residual:.1e}',
            }
            if step_dt != self.params.dt:
                extra['dt'] = f'{step_dt:.1e}'
            pbar.set_extra(extra)
            pbar.step(1)

        pbar.end()

        # Cleanup
        snes.destroy()

        if rank == 0:
            print("\nDynamic simulation finished.")

    def compute_energy(self, U_func):
        """
        Compute the total free energy and its individual contributions.

        Parameters
        ----------
        U_func : fem.Function
            Current Q-tensor solution.

        Returns
        -------
        dict
            Keys: 'total', 'elastic', 'surf_dome', 'surf_base' - all in Joules.
        """
        Q   = _make_Q(U_func)
        p   = self.params
        ID_CAP    = self.mesh_handler.ID_CAP
        ID_DOME   = self.mesh_handler.ID_DOME
        ID_BOTTOM = self.mesh_handler.ID_BOTTOM

        # Elastic
        dQ = grad(Q)
        dQ_swap = as_tensor([[[dQ[i, k, j] for k in range(3)]
                               for j in range(3)]
                              for i in range(3)])
        f_elastic = (0.5 * p.L1 * inner(dQ, dQ)
                   + 0.5 * p.L2 * inner(div(Q), div(Q))
                   + 0.5 * p.L3 * inner(dQ_swap, dQ))
        E_elastic = MPI.COMM_WORLD.allreduce(
            _fem.assemble_scalar(_fem.form(f_elastic * self.dx(ID_CAP))), op=MPI.SUM)

        # Dome anchoring: planar degenerate
        nQn = inner(outer(self.n_dome, self.n_dome), Q)
        f_dome = 0.5 * p.C_surface * (nQn + p.S / 3.0) ** 2
        E_dome = MPI.COMM_WORLD.allreduce(
            _fem.assemble_scalar(_fem.form(f_dome * self.ds(ID_DOME))), op=MPI.SUM)

        # Base anchoring: Rapini-Papoular (in-plane) + planar degenerate (tilt)
        dQ_base = Q - self.Q_s_base
        nQn_b = inner(outer(self.n_base, self.n_base), Q)
        f_base = (0.5 * p.C_polyamide * inner(dQ_base, dQ_base)
                + 0.5 * p.C_surface * (nQn_b + p.S / 3.0) ** 2)
        E_base = MPI.COMM_WORLD.allreduce(
            _fem.assemble_scalar(_fem.form(f_base * self.ds(ID_BOTTOM))), op=MPI.SUM)

        E_total = E_elastic + E_dome + E_base
        return {'total': E_total, 'elastic': E_elastic,
                'surf_dome': E_dome, 'surf_base': E_base}

    def close(self):
        if MPI.COMM_WORLD.rank == 0 and self.csv_file is not None:
            try:
                self.csv_file.close()
            except:
                pass


def _write_params_txt(p):
    """Write a plain-text copy of all simulation parameters to the output directory."""
    lines = []

    lines += [
        "SCALAR ORDER PARAMETER",
        f"  S                   = {p.S}",
        "",
        "FRANK ELASTIC CONSTANTS",
        f"  K1                  = {p.K1:.3e}  N  (splay)",
        f"  K2                  = {p.K2:.3e}  N  (twist)",
        f"  K3                  = {p.K3:.3e}  N  (bend)",
        "",
        "LdG ELASTIC CONSTANTS",
        f"  L1                  = {p.L1:.6e}  N",
        f"  L2                  = {p.L2:.6e}  N",
        f"  L3                  = {p.L3:.6e}  N",
        "",
        "SURFACE ANCHORING",
        f"  C_surface           = {p.C_surface:.3e}  J/m^2",
        f"  C_polyamide         = {p.C_polyamide:.3e}  J/m^2",
        "",
        "BASE RUBBING DIRECTION",
        f"  D_rub               = {list(p.D_rub)}",
        "",
        "DROPLET GEOMETRY",
        f"  base_radius         = {p.base_radius:.3e}  m",
        f"  contact_angle_deg   = {p.contact_angle_deg}  degrees",
        "",
        "MESH PARAMETERS",
        f"  mesh_size_factor    = {p.mesh_size_factor}",
        f"  mesh_size           = {p.mesh_size:.3e}  m",
        f"  min_bulk_layers     = {p.min_bulk_layers}",
        "",
        "DYNAMICS",
        f"  gamma_viscosity     = {p.gamma_viscosity}  Pa.s",
        f"  dt                  = {p.dt}  s",
        f"  num_steps           = {p.num_steps}",
        f"  t_final             = {p.dt * p.num_steps:.3e}  s",
        "",
        "SOLVER",
        f"  newton_max_iter     = {p.newton_max_iter}",
        f"  newton_tol          = {p.newton_tol:.3e}",
        "",
        "INITIAL CONDITION",
        f"  initial_condition_type = {p.initial_condition_type!r}",
        f"  ic_xdmf_file           = {p.ic_xdmf_file!r}",
        "",
        "OUTPUT",
        f"  output_dir              = {p.output_dir!r}",
        f"  output_every_n_steps    = {p.output_every_n_steps}",
    ]

    path = os.path.join(p.output_dir, "parameters.txt")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Parameters saved to: {path}")


def run_simulation(sim_params):
    """
    Orchestrate the full Nematic LC simulation workflow (stages 1–6).

    Parameters
    ----------
    sim_params : SimulationParameters
        Fully configured parameter object from params_N.py.

    Returns
    -------
    str
        Absolute path to the output directory containing XDMF and CSV files.
    """
    from .meshing import MeshHandler, FunctionSpaces
    from .initial_conditions import InitialConditions
    from .boundary_conditions import BoundaryConditions
    from .output_handler import OutputHandler

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n" + "=" * 70)
        print(" FEniCSx Nematic LC Ginzburg-Landau Dynamics Solver")
        print("=" * 70 + "\n")
        print(f"  Droplet diameter: {sim_params.base_radius * 2e6:.1f} um")
        print(f"  Mesh size factor: {sim_params.mesh_size_factor}")
        print(f"  Time step: {sim_params.dt:.1e} s, Total steps: {sim_params.num_steps}")
        print(f"\n  Q-tensor parameters:")
        print(f"    S  = {sim_params.S:.4f}  (scalar order parameter)")
        print(f"    K1 = {sim_params.K1:.3e} N  K2 = {sim_params.K2:.3e} N  K3 = {sim_params.K3:.3e} N")
        print(f"    L1 = {sim_params.L1:.3e} N  L2 = {sim_params.L2:.3e} N  L3 = {sim_params.L3:.3e} N")

    comm.Barrier()

    solver = None
    output_handler = None
    try:
        # 1. Setup
        mesh_handler = MeshHandler(sim_params)
        mesh_handler.create_mesh()
        spaces = FunctionSpaces(mesh_handler)
        spaces.setup_spaces()

        # 2. Create functions (BDF2 needs two previous states)
        U_current = _fem.Function(spaces.W, name="U_current")
        U_previous = _fem.Function(spaces.W, name="U_previous")
        U_prev_prev = _fem.Function(spaces.W, name="U_prev_prev")

        # 3. Initial Conditions
        initial_cond = InitialConditions(sim_params, spaces)
        initial_cond.set_initial_state(U_current)
        U_previous.x.array[:] = U_current.x.array
        U_prev_prev.x.array[:] = U_current.x.array

        # 4. Boundary Conditions
        bcs_handler = BoundaryConditions(sim_params, spaces, mesh_handler)
        bcs_list = bcs_handler.apply_all_bcs()

        # 5. Output Setup
        output_handler = OutputHandler(sim_params, mesh_handler)
        output_handler.save_mesh_and_tags()
        if rank == 0:
            _write_params_txt(sim_params)

        # 6. Solver Setup
        solver = GinzburgLandauSolver(sim_params, spaces, mesh_handler, bcs_list, output_handler)

        E_dict_initial = solver.compute_energy(U_current)
        E_initial = E_dict_initial['total']
        if rank == 0:
            print(f"\nInitial energy: {E_initial:.6e} J")

        # 7. Run Simulation
        solver.run_time_stepping(U_current, U_previous, U_prev_prev)

        # 8. Final State
        E_dict_final = solver.compute_energy(U_current)
        E_final = E_dict_final['total']
        if rank == 0:
            print(f"\nFinal energy (t={sim_params.num_steps * sim_params.dt:.1e} s): {E_final:.6e} J")
            print(f"Total reduction: {E_initial - E_final:.6e} J")
            print(f"\nTime-series output in: {sim_params.output_dir}/")

        return sim_params.output_dir

    finally:
        if solver is not None:
            solver.close()
        if output_handler is not None:
            output_handler.close()
        comm.Barrier()
