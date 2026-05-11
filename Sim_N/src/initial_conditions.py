"""
src/initial_conditions.py

Initial Q-tensor field setup for the Nematic LC simulation.

The Q-tensor is stored as a 5-component vector [q0, q1, q2, q3, q4] encoding
the symmetric traceless 3×3 tensor:

    Q = [[ q0,   q2,   q3  ],
         [ q2,   q1,   q4  ],
         [ q3,   q4,  -q0-q1]]

For a uniaxial director n with scalar order parameter S:
    Q = S · (n⊗n − I/3)

Functions
---------
_n_to_q5
    Convert a unit director (nx, ny, nz) and S into the 5-component vector.

Classes
-------
InitialConditions
    Populates a DOLFINx Function with one of three initial director fields:
    'random', 'radial', or 'from_file'.
"""

import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Helper: director → 5-component Q vector
# ---------------------------------------------------------------------------

def _n_to_q5(nx, ny, nz, S):
    """
    Convert a unit director and scalar order parameter to the 5-component Q vector.

    Parameters
    ----------
    nx, ny, nz : float
        Components of the unit director (|n| = 1 assumed).
    S : float
        Scalar nematic order parameter.

    Returns
    -------
    tuple[float, float, float, float, float]
        (q0, q1, q2, q3, q4) encoding Q = S·(n⊗n − I/3).
    """
    q0 = S * (nx*nx - 1.0/3.0)
    q1 = S * (ny*ny - 1.0/3.0)
    q2 = S * nx * ny
    q3 = S * nx * nz
    q4 = S * ny * nz
    return q0, q1, q2, q3, q4


# ---------------------------------------------------------------------------
# Initial condition dispatcher
# ---------------------------------------------------------------------------

class InitialConditions:
    """
    Populates the Q-tensor Function with a chosen initial director field.

    In all modes the dome rim nodes (r ≈ R_base) are set to the azimuthal
    director n_az = (−y/r, x/r, 0) so the IC already satisfies the hard
    Dirichlet BC applied there by BoundaryConditions.

    Parameters
    ----------
    params : SimulationParameters
        Simulation parameters (uses S, base_radius, contact_angle_rad,
        initial_condition_type, ic_xdmf_file).
    spaces : FunctionSpaces
        Function space container (uses W for DOF coordinate lookup).
    """

    def __init__(self, params, spaces):
        self.params = params
        self.spaces = spaces

    def set_initial_state(self, U_func):
        """
        Dispatch to the appropriate IC setter based on params.initial_condition_type.

        Parameters
        ----------
        U_func : fem.Function
            Q-tensor Function to be overwritten with the initial field.

        Raises
        ------
        ValueError
            If params.initial_condition_type is not 'random', 'radial', or 'from_file'.
        """
        ic_type = self.params.initial_condition_type.lower()
        if ic_type == 'random':
            self._set_random_initial_state(U_func)
        elif ic_type == 'radial':
            self._set_radial_initial_state(U_func)
        elif ic_type == 'from_file':
            self._set_from_file_initial_state(U_func)
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type!r}. "
                             f"Use 'random', 'radial', or 'from_file'.")

    # -----------------------------------------------------------------------
    # IC: random director
    # -----------------------------------------------------------------------

    def _set_random_initial_state(self, U_func):
        """
        Assign a random unit director at every node.

        Rim nodes are forced to the azimuthal director (nz = 0) so the IC
        already satisfies the hard q3 = q4 = 0 constraint applied at the rim.

        Parameters
        ----------
        U_func : fem.Function
            Q-tensor Function to populate.
        """
        if MPI.COMM_WORLD.rank == 0:
            print("InitialConditions: Setting random Q-tensor IC...")

        S      = self.params.S
        R_base = self.params.base_radius

        def random_Q(x):
            num_points = x.shape[1]
            q = np.zeros((5, num_points))
            for i in range(num_points):
                rx, ry = x[0, i], x[1, i]
                r_xy = np.sqrt(rx**2 + ry**2)
                if np.isclose(r_xy, R_base, rtol=1e-4):
                    # Rim: azimuthal director, nz = 0 (matches hard BC)
                    nx, ny, nz = -ry / (r_xy + 1e-30), rx / (r_xy + 1e-30), 0.0
                    q[0,i], q[1,i], q[2,i], q[3,i], q[4,i] = _n_to_q5(nx, ny, nz, S)
                else:
                    v = np.random.randn(3)
                    v /= np.linalg.norm(v) + 1e-10
                    q[0,i], q[1,i], q[2,i], q[3,i], q[4,i] = _n_to_q5(v[0], v[1], v[2], S)
            return q

        U_func.interpolate(random_Q)
        U_func.x.scatter_forward()

    # -----------------------------------------------------------------------
    # IC: radial (spherical outward) director
    # -----------------------------------------------------------------------

    def _set_radial_initial_state(self, U_func):
        """
        Assign a radially outward director field (spherical outward normal).

        This gives a hedgehog-like IC close to the radial configuration expected
        for strong homeotropic anchoring, and converges faster than random when
        the final state is radial.  Exceptions:
          - Rim nodes → azimuthal director (hard BC compatibility).
          - Base nodes → uniform +x director (avoids singularity at r = 0).

        Parameters
        ----------
        U_func : fem.Function
            Q-tensor Function to populate.
        """
        if MPI.COMM_WORLD.rank == 0:
            print("InitialConditions: Setting radial Q-tensor IC (spherical geometry)...")

        S = self.params.S
        p = self.params
        R_base   = p.base_radius
        theta    = p.contact_angle_rad
        R_sphere = R_base / np.sin(theta)
        z_center = -R_sphere * np.cos(theta)

        def radial_Q(x):
            num_points = x.shape[1]
            q = np.zeros((5, num_points))
            for i in range(num_points):
                rx, ry, rz = x[0, i], x[1, i], x[2, i]
                r_xy = np.sqrt(rx**2 + ry**2)
                if np.isclose(r_xy, R_base, rtol=1e-4):
                    # Dome rim: azimuthal director (hard BC compatibility)
                    nx, ny, nz = -ry / (r_xy + 1e-30), rx / (r_xy + 1e-30), 0.0
                elif np.isclose(rz, 0.0, atol=1e-10):
                    # Base: uniform +x to avoid the polar singularity at r = 0
                    nx, ny, nz = 1.0, 0.0, 0.0
                else:
                    # Bulk / dome interior: spherical outward normal from sphere centre
                    nx = rx
                    ny = ry
                    nz = rz - z_center
                    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
                    nx, ny, nz = nx/mag, ny/mag, nz/mag
                q[0,i], q[1,i], q[2,i], q[3,i], q[4,i] = _n_to_q5(nx, ny, nz, S)
            return q

        U_func.interpolate(radial_Q)
        U_func.x.scatter_forward()

    # -----------------------------------------------------------------------
    # IC: load from previous simulation file
    # -----------------------------------------------------------------------

    def _set_from_file_initial_state(self, U_func):
        """
        Load the last saved Q-tensor frame from an HDF5 file and interpolate
        it onto the current simulation mesh.

        Reads geometry and Q_tensor data from the .h5 companion file of the
        XDMF output.  Uses LinearNDInterpolator (scipy) for mesh-to-mesh
        interpolation, which handles non-matching meshes robustly.  DOFs that
        fall outside the source mesh convex hull are set to zero.

        Parameters
        ----------
        U_func : fem.Function
            Q-tensor Function to populate.

        Raises
        ------
        ValueError
            If params.ic_xdmf_file is None.
        RuntimeError
            If the mesh geometry or Q_tensor data cannot be found in the HDF5 file.
        """
        import h5py
        import os
        from scipy.interpolate import LinearNDInterpolator

        comm = MPI.COMM_WORLD
        rank = comm.rank

        xdmf_path = self.params.ic_xdmf_file
        if xdmf_path is None:
            raise ValueError("params.ic_xdmf_file must be set when using ic_type='from_file'")

        h5_path = os.path.splitext(xdmf_path)[0] + '.h5'

        if rank == 0:
            print(f"InitialConditions: Loading last frame from '{h5_path}'...")

        # -----------------------------------------------------------------------
        # STAGE 1 - Read mesh coordinates and Q_tensor from HDF5
        # -----------------------------------------------------------------------
        # Try several known geometry path conventions produced by different
        # DOLFINx / meshio versions so the loader is robust to renamed datasets.
        with h5py.File(h5_path, 'r') as h5f:
            possible_paths = [
                'Mesh/Cap_Mesh/geometry',
                'Mesh/Loaded_Mesh/geometry',
                'Mesh/mesh/geometry',
                'Mesh/geometry',
                'geometry',
            ]
            mesh_coords = None
            for path in possible_paths:
                if path in h5f:
                    mesh_coords = h5f[path][:]
                    if rank == 0:
                        print(f"  Found mesh geometry at: {path}")
                    break
            if mesh_coords is None:
                raise RuntimeError(f"Mesh coordinates not found in '{h5_path}'")

            if 'Function' not in h5f or 'Q_tensor' not in h5f['Function']:
                raise RuntimeError(f"'Function/Q_tensor' not found in '{h5_path}'")

            # Load the chronologically last snapshot (keys are zero-padded integers)
            timestep_keys = sorted(h5f['Function/Q_tensor'].keys())
            last_key = timestep_keys[-1]
            if rank == 0:
                print(f"  Found {len(timestep_keys)} timesteps; loading key '{last_key}'")

            Q_data = h5f[f'Function/Q_tensor/{last_key}'][:]

        # -----------------------------------------------------------------------
        # STAGE 2 - Build mesh-to-mesh interpolators
        # -----------------------------------------------------------------------
        # Normalise to (n_nodes, 5) in case the dataset is stored flattened
        if Q_data.ndim == 1:
            Q_data = Q_data.reshape(-1, 5)

        n_pts = min(mesh_coords.shape[0], Q_data.shape[0])
        mesh_coords = mesh_coords[:n_pts]
        Q_data      = Q_data[:n_pts]

        if rank == 0:
            print("  Building LinearNDInterpolator (5 components)...")

        interps = [LinearNDInterpolator(mesh_coords, Q_data[:, c]) for c in range(5)]

        # -----------------------------------------------------------------------
        # STAGE 3 - Evaluate at current mesh DOF coordinates and assemble
        # -----------------------------------------------------------------------
        dof_coords = self.spaces.W.tabulate_dof_coordinates()

        Qc = [interp(dof_coords) for interp in interps]

        # DOFs outside the source mesh convex hull return NaN; reset to zero
        # so SNES has a well-defined starting point even at domain edges.
        nan_mask = np.any(np.isnan(np.stack(Qc, axis=1)), axis=1)
        for c in range(5):
            Qc[c][nan_mask] = 0.0

        if rank == 0 and np.any(nan_mask):
            print(f"  Warning: {nan_mask.sum()} DOFs outside source mesh - set to zero.")

        # DOLFINx stores components interleaved: [q0_0, q1_0, ..., q4_0, q0_1, ...]
        n_nodes = dof_coords.shape[0]
        data = np.empty(n_nodes * 5, dtype=np.float64)
        for c in range(5):
            data[c::5] = Qc[c]

        U_func.x.array[:] = data
        U_func.x.scatter_forward()

        if rank == 0:
            print("  Interpolation onto simulation mesh complete.")
