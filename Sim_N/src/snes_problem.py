"""
src/snes_problem.py

PETSc SNES callback wrapper for the monolithic Q-tensor system.

Classes
-------
SNESProblem
    Wraps UFL residual and Jacobian forms into the PETSc SNES callback
    interface (F, J methods).  Vectors and matrices are pre-allocated once
    and reused across all Newton iterations to avoid repeated allocation.
"""

from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# SNES callback wrapper
# ---------------------------------------------------------------------------

class SNESProblem:
    """
    Nonlinear problem class compatible with the PETSc SNES solver.

    Assembles the residual F(x) and Jacobian J(x) from pre-compiled UFL forms.
    Pre-allocating the PETSc vector and matrix in __init__ avoids repeated
    memory allocation during the Newton iteration loop.

    Parameters
    ----------
    F_form : ufl.Form
        Residual weak form.
    J_form : ufl.Form
        Jacobian weak form (derivative of F_form w.r.t. the solution).
    u : fem.Function
        Solution function living in the mixed Q-tensor space.
    bcs : list[fem.DirichletBC]
        Dirichlet boundary conditions to lift and apply during assembly.
    """

    def __init__(self, F_form, J_form, u, bcs):
        self._F   = fem.form(F_form)
        self._J   = fem.form(J_form)
        self._u   = u
        self._bcs = bcs
        # Pre-allocate once; reused every SNES iteration to avoid malloc overhead
        self._b = dolfinx.fem.petsc.create_vector(self._u.function_space)
        self._A = dolfinx.fem.petsc.create_matrix(self._J)

    @property
    def b(self):
        return self._b

    @property
    def A(self):
        return self._A

    def obj(self, snes: PETSc.SNES, x: PETSc.Vec):
        """
        Compute the objective (residual norm) for SNES line-search.

        Parameters
        ----------
        snes : PETSc.SNES
            The SNES solver instance (passed by PETSc internally).
        x : PETSc.Vec
            Current iterate.

        Returns
        -------
        float
            Euclidean norm of F(x).
        """
        self.F(snes, x, self._b)
        return self._b.norm()

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """
        Assemble the residual vector F(x) into b.

        Parameters
        ----------
        snes : PETSc.SNES
            The SNES solver instance.
        x : PETSc.Vec
            Current iterate to evaluate F at.
        b : PETSc.Vec
            Output vector; overwritten with F(x).
        """
        # Copy the SNES iterate into the DOLFINx Function so UFL forms see
        # the latest values (ghost update ensures off-process DOFs are correct)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self._u.x.petsc_vec)
        self._u.x.scatter_forward()

        # Zero, assemble, lift Dirichlet BCs, and scatter
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self._F)
        dolfinx.fem.petsc.apply_lifting(b, [self._J], bcs=[self._bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self._bcs, x, -1.0)

    def J(self, snes: PETSc.SNES, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """
        Assemble the Jacobian matrix J(x) into A.

        Parameters
        ----------
        snes : PETSc.SNES
            The SNES solver instance.
        x : PETSc.Vec
            Current iterate.
        A : PETSc.Mat
            Output matrix; overwritten with J(x).
        P : PETSc.Mat
            Preconditioner matrix (same as A for direct solvers).
        """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self._J, bcs=self._bcs)
        A.assemble()
