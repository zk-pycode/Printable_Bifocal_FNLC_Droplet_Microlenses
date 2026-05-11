"""
src/boundary_conditions.py

Hard and soft boundary conditions for the Q-tensor simulation.

Hard Dirichlet BCs (enforced via DOLFINx DirichletBC):
    Dome rim (r = R_max, top contact line): all 5 Q components locked to the
    azimuthal director n_az = (−y/r, x/r, 0), giving a ring of tangentially
    aligned LC that anchors the twist sense around the droplet perimeter.

Soft BCs (penalty terms in the weak form - see solver.py):
    Base z=0:  C_surface/2   · (ẑ^T Q ẑ + S/3)²  [planar degenerate, suppresses tilt]
               C_polyamide/2 · |Q − Q_s_base|²     [Rapini-Papoular toward rubbing axis]
    Dome:      C_surface/2 · (ν^T Q ν + S/3)²  [planar degenerate, tangential to dome]

Classes
-------
BoundaryConditions
    Builds and returns the list of hard DirichletBC objects for the SNES solve.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem as _fem


# ---------------------------------------------------------------------------
# Boundary condition builder
# ---------------------------------------------------------------------------

class BoundaryConditions:
    """
    Builds hard Dirichlet boundary conditions for the Q-tensor system.

    Only the dome rim receives hard BCs; all other surfaces are handled by
    soft penalty terms assembled directly into the weak form in solver.py.

    Parameters
    ----------
    params : SimulationParameters
        Simulation parameters (uses S, base_radius).
    spaces : FunctionSpaces
        Function space container (uses W, the mixed Q-tensor space).
    mesh_handler : MeshHandler
        Mesh container (uses mesh geometry to locate rim nodes).
    """

    def __init__(self, params, spaces, mesh_handler):
        self.params       = params
        self.spaces       = spaces
        self.mesh_handler = mesh_handler

    def apply_all_bcs(self):
        """
        Locate dome-rim nodes and build DirichletBC objects for all 5 Q components.

        The rim is identified geometrically as nodes at r = R_max (the outermost
        radial extent of the mesh).  Each component is constrained separately
        because DOLFINx DirichletBC operates on sub-spaces.

        Returns
        -------
        list[fem.DirichletBC]
            Five DirichletBC objects (one per Q component) locking the rim to
            the azimuthal director field.
        """
        comm = MPI.COMM_WORLD
        mesh = self.mesh_handler.mesh
        W    = self.spaces.W
        S    = self.params.S

        # -----------------------------------------------------------------------
        # Locate the dome rim - the largest radial extent of the mesh
        # -----------------------------------------------------------------------
        # R_max is computed globally across all MPI ranks so every rank agrees
        # on the threshold used by rim_marker.
        coords = mesh.geometry.x
        r_all  = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        R_max  = comm.allreduce(float(np.max(r_all)) if r_all.size > 0 else 0.0,
                                op=MPI.MAX)

        if comm.rank == 0:
            print(f"  Dome rim radius: R_max = {R_max*1e6:.3f} um")

        def rim_marker(x):
            return np.isclose(np.sqrt(x[0]**2 + x[1]**2), R_max, rtol=1e-4)

        # -----------------------------------------------------------------------
        # Build Q-tensor values at rim nodes: azimuthal director n_az = (−y/r, x/r, 0)
        # -----------------------------------------------------------------------
        def make_q_comp(comp, S):
            # Returns a lambda that evaluates one scalar Q component at rim points.
            # The azimuthal director has nz = 0, so q3 = Q_13 = 0 and q4 = Q_23 = 0,
            # which is consistent with the planar base anchoring.
            def q_scalar(x):
                vals = np.zeros(x.shape[1])
                for i in range(x.shape[1]):
                    rx, ry = x[0, i], x[1, i]
                    r_xy = np.sqrt(rx**2 + ry**2) + 1e-30
                    nx, ny = -ry / r_xy, rx / r_xy
                    q_vals = [
                        S * (nx*nx - 1.0/3.0),  # q0 = Q_11
                        S * (ny*ny - 1.0/3.0),  # q1 = Q_22
                        S * nx * ny,             # q2 = Q_12
                        0.0,                     # q3 = Q_13  (nz = 0)
                        0.0,                     # q4 = Q_23  (nz = 0)
                    ]
                    vals[i] = q_vals[comp]
                return vals
            return q_scalar

        # -----------------------------------------------------------------------
        # Create one DirichletBC per Q component
        # -----------------------------------------------------------------------
        bcs = []
        for comp in range(5):
            V_sub, _ = W.sub(comp).collapse()
            dofs = _fem.locate_dofs_geometrical((W.sub(comp), V_sub), rim_marker)
            g = _fem.Function(V_sub)
            g.interpolate(make_q_comp(comp, S))
            bcs.append(_fem.dirichletbc(g, dofs, W.sub(comp)))

        if comm.rank == 0:
            print("  Hard BCs: all 5 Q components locked at dome rim (azimuthal director).")
            print("  Base and dome surface anchoring enforced as soft penalties in solver.py.")

        return bcs
