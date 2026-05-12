"""
src/output_handler.py

XDMF output for the director field extracted from the Q-tensor solution.

At each saved time step the 5-component Q-tensor is diagonalised node-by-node
and the eigenvector with the largest eigenvalue is written as the director
n = (nx, ny, nz).  A temporal sign-consistency fix prevents 180° director
flips between frames that would otherwise appear as noise in ParaView.

Functions
---------
_extract_director
    Eigendecompose Q at every owned node and return the principal eigenvector.

Classes
-------
OutputHandler
    Opens, writes, and closes the XDMF/HDF5 director time-series file.
    Also writes the mesh and boundary-tag file once at startup.
"""

import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem as _fem, io
from basix.ufl import element


# ---------------------------------------------------------------------------
# Helper: Q-tensor → director field
# ---------------------------------------------------------------------------

def _extract_director(U_func):
    """
    Extract the principal director from a 5-component Q-tensor Function.

    The Q-tensor is reconstructed from the 5-component storage format:

        Q = [[ q0,   q2,   q3  ],
             [ q2,   q1,   q4  ],
             [ q3,   q4,  -q0-q1]]

    The director is the eigenvector of Q corresponding to the largest
    eigenvalue (the "long axis" of the nematic alignment ellipsoid).
    numpy.linalg.eigh returns eigenvalues in ascending order, so the
    director is the last column of the eigenvector matrix.

    Parameters
    ----------
    U_func : fem.Function
        Q-tensor Function with 5-component (block) DOFs.

    Returns
    -------
    np.ndarray, shape (n_owned_nodes, 3)
        Unit director at each locally owned mesh node.
    """
    bs      = U_func.function_space.dofmap.index_map_bs   # = 5
    n_owned = U_func.function_space.dofmap.index_map.size_local
    Q5      = U_func.x.array[:n_owned * bs].reshape(n_owned, bs)

    Q_mat          = np.zeros((n_owned, 3, 3))
    Q_mat[:, 0, 0] = Q5[:, 0]
    Q_mat[:, 1, 1] = Q5[:, 1]
    Q_mat[:, 2, 2] = -Q5[:, 0] - Q5[:, 1]
    Q_mat[:, 0, 1] = Q_mat[:, 1, 0] = Q5[:, 2]
    Q_mat[:, 0, 2] = Q_mat[:, 2, 0] = Q5[:, 3]
    Q_mat[:, 1, 2] = Q_mat[:, 2, 1] = Q5[:, 4]

    # eigh guarantees real eigenvalues for symmetric matrices and is faster
    # than eig; eigenvectors are returned as columns, sorted ascending by eigenvalue
    _, evecs = np.linalg.eigh(Q_mat)   # evecs: (n, 3, 3)
    return evecs[:, :, -1]             # largest eigenvalue → last column


# ---------------------------------------------------------------------------
# Output handler
# ---------------------------------------------------------------------------

class OutputHandler:
    """
    Manages XDMF/HDF5 output for the director time-series.

    Writes the mesh and boundary tags once at startup, then appends the
    director field at each requested time step.  A per-node sign-consistency
    check between consecutive frames prevents the arbitrary ±1 eigenvector
    sign flip from appearing as sudden director reversals in animations.

    Parameters
    ----------
    params : SimulationParameters
        Simulation parameters (uses output_dir).
    mesh_handler : MeshHandler
        Mesh container (uses mesh, cell_tags, boundary_tags).
    """

    def __init__(self, params, mesh_handler):
        self.params       = params
        self.mesh_handler = mesh_handler
        self.comm         = MPI.COMM_WORLD
        self.xdmf_file    = None
        self._n_func      = None
        self._prev_directors = None   # stored for inter-frame sign consistency

        if self.comm.rank == 0 and not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)
        self.comm.Barrier()

    def save_mesh_and_tags(self):
        """
        Write the mesh geometry, cell tags, and boundary tags to mesh.xdmf.

        Called once before the time loop so ParaView can load the mesh
        independently of the director time-series file.
        """
        msh      = self.mesh_handler.mesh
        filepath = os.path.join(self.params.output_dir, "mesh.xdmf")
        try:
            with io.XDMFFile(self.comm, filepath, "w",
                             encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
                xdmf.write_mesh(msh)
                xdmf.write_meshtags(self.mesh_handler.cell_tags,     msh.geometry)
                xdmf.write_meshtags(self.mesh_handler.boundary_tags, msh.geometry)
        except Exception as e:
            if self.comm.rank == 0:
                print(f"ERROR saving mesh: {e}")
        self.comm.Barrier()

    def setup_output_files(self, spaces):
        """
        Create the director Function and open the XDMF time-series file.

        A separate CG1 vector space (3 components) is created for the director
        because the simulation operates in the 5-component Q-tensor space.

        Parameters
        ----------
        spaces : FunctionSpaces
            Function space container (mesh is read from spaces.mesh indirectly
            via mesh_handler; argument kept for API consistency).

        Returns
        -------
        fem.Function
            The director Function (also stored as self._n_func).
        """
        if self.comm.rank == 0:
            print("OutputHandler: Setting up director time-series XDMF file...")

        msh   = self.mesh_handler.mesh
        n_el  = element("Lagrange", msh.basix_cell(), 1, shape=(3,))
        V_dir = _fem.functionspace(msh, n_el)
        self._n_func = _fem.Function(V_dir, name="Director")

        filepath = os.path.join(self.params.output_dir, "simulation_n.xdmf")
        try:
            self.xdmf_file = io.XDMFFile(self.comm, filepath, "w",
                                          encoding=io.XDMFFile.Encoding.HDF5)
            self.xdmf_file.write_mesh(msh)
        except Exception as e:
            if self.comm.rank == 0:
                print(f"ERROR opening director XDMF file: {e}")
            self.comm.Abort(1)

        return self._n_func

    def write_timestep(self, _unused, U_current, t):
        """
        Extract the director from the Q-tensor and append it to the XDMF file.

        Applies a greedy per-node sign correction so that each director points
        into the same hemisphere as it did in the previous frame.  This prevents
        the arbitrary ±1 eigenvector sign returned by eigh from producing
        spurious 180° flips in ParaView time animations.

        Parameters
        ----------
        _unused : object
            Placeholder (not used; solver passes the director Function here for
            API compatibility, but the director is recomputed from U_current).
        U_current : fem.Function
            Current Q-tensor solution Function.
        t : float
            Current simulation time in seconds.
        """
        if self.xdmf_file is None:
            if self.comm.rank == 0:
                print("ERROR: write_timestep called before setup_output_files.")
            return

        try:
            n_owned   = self._n_func.function_space.dofmap.index_map.size_local
            directors = _extract_director(U_current)   # (n_owned, 3)

            # Flip any node whose director has reversed relative to the last frame
            # (dot product < 0 indicates a sign flip from eigh's arbitrary choice)
            if self._prev_directors is not None:
                flip = np.einsum('ij,ij->i', directors, self._prev_directors) < 0.0
                directors[flip] *= -1.0
            self._prev_directors = directors.copy()

            self._n_func.x.array[:n_owned * 3] = directors.reshape(-1)
            self._n_func.x.scatter_forward()
            self.xdmf_file.write_function(self._n_func, t)
        except Exception as e:
            if self.comm.rank == 0:
                print(f"ERROR writing timestep t={t}: {e}")

    def close(self):
        """Close the XDMF file and synchronise all MPI ranks."""
        if self.xdmf_file is not None:
            try:
                self.xdmf_file.close()
            except Exception as e:
                if self.comm.rank == 0:
                    print(f"ERROR closing director XDMF file: {e}")
        self.comm.Barrier()
        
