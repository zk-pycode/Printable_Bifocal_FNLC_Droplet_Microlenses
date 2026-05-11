#  Ginzburg-Landau Q-tensor Based Nematic Liquid Crystal Droplet Simulation 

## Introduction

This code simulates the orientational dynamics of a nematic liquid crystal (LC) confined inside a spherical-cap droplet sitting on a substrate. The director field is evolved (free energy minimization) using the Q-tensor, discretised with a mixed finite element method on a tetrahedral mesh of the droplet volume. The simulation accounts for three Frank elastic constants (splay, twist, bend) mapped to the LdG framework, planar degenerate anchoring at the dome surface, and Rapini-Papoular anchoring at the polyamide substrate.

## Code Overview
```bash
Sim_N/
├── main_N.py                   Entry point - loads parameters and calls run_simulation
├── params_N.py                 All tunable settings (edit this file)
├── README.md                   This file
└── src/
    ├── __init__.py
    ├── solver.py               GinzburgLandauSolver (weak form, BDF1/BDF2 time stepping, energy logging)
    │                           run_simulation (full pipeline orchestration)
    ├── meshing.py              MeshHandler (spherical-cap tet mesh via gmsh + prism tet decomposition)
    │                           FunctionSpaces (5-component CG1 Q-tensor space)
    ├── initial_conditions.py   InitialConditions (random, radial, or loaded from a previous file)
    ├── boundary_conditions.py  BoundaryConditions (hard Dirichlet BCs at the dome rim)
    ├── snes_problem.py         SNESProblem (PETSc SNES callback wrapper for the monolithic system)
    └── output_handler.py       OutputHandler (XDMF director time-series and energy CSV)
```

## Physics

The Q-tensor obeys the Ginzburg-Landau relaxation equation

```
gamma * dQ/dt = -dF/dQ
```

with the free energy functional

```
F = integral_volume [
        L1/2 * |grad(Q)|**2
      + L2/2 * |div(Q)|**2
      + L3/2 * Q[i,j,k] * Q[i,k,j]
      ] dV

  + integral_dome [
        C_surface/2 * (nu.T @ Q @ nu + S/3)**2
      ] dS

  + integral_base [
        C_polyamide/2 * |Q - Q_s|**2
      + C_surface/2  * (z.T @ Q @ z + S/3)**2
      ] dS
```

The elastic constants L1, L2, L3 are derived from the measured Frank constants K1 (splay), K2 (twist), K3 (bend) via the standard LdG–Frank mapping at scalar order parameter S.

## Environment

The code requires **DOLFINx v0.9.0** (FEniCSx) with PETSc compiled with MUMPS support, plus the following Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `dolfinx` | 0.9.0 | FEM assembly and mesh distribution |
| `gmsh` | 4.15.0 | 2D base-disk meshing |
| `petsc4py` | 3.24.0 | SNES Newton solver |
| `mpi4py` | 4.1.1 | MPI parallelism |
| `numpy` | 2.3.1 | Numerics and mesh-to-mesh interpolation |
| `scipy` | 1.16.3 | Mesh-to-mesh interpolation |
| `h5py` | 3.15.1 | Reading previous simulation files (`from_file` IC) |
| `progiter` | 2.0.0 | Progress reporting |

### Installing DOLFINx

Detailed installation instructions for DOLFINx can be found at [FEniCS/dolfinx](https://github.com/FEniCS/dolfinx). The recommended approach on Linux is via conda.


## Running the Simulation

1. **Edit parameters** in `params_N.py` — set the droplet geometry, elastic constants, anchoring strengths, time step, and output directory.

2. **Run** (serial or parallel):
   ```bash
   python main_N.py
   # or in parallel (recommended for large meshes):
   mpirun -n 4 python main_N.py
   ```

3. **Monitor progress** — the solver prints per-step SNES status, residual norm, and total energy to stdout. An energy time-series is also written to `<output_dir>/dynamic_log.csv`.

## Key Parameters (`params_N.py`)

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `S` | Scalar nematic order parameter | `0.5` |
| `K1, K2, K3` | Frank elastic constants (N) | `0.4–2 × 10⁻¹²` |
| `C_surface` | Planar-degenerate anchoring strength on dome and base tilt (J/m²) | `1 × 10⁻⁶` |
| `C_polyamide` | Base Rapini-Papoular anchoring toward rubbing direction (J/m²) | `2 × 10⁻⁶` |
| `base_radius` | Droplet base radius (m) | `127 × 10⁻⁶` |
| `contact_angle_deg` | Static contact angle (degrees) | `20–30` |
| `gamma_viscosity` | Rotational viscosity γ₁ (Pa·s) | `0.06` |
| `dt` | Time step (s) | `0.01` |
| `initial_condition_type` | `'random'`, `'radial'`, or `'from_file'` | `'random'` |

## Output
```bash
<output_dir>/
├── simulation_n.xdmf           Director field time-series (open in ParaView)
├── simulation_n.h5             HDF5 data store for the XDMF time-series
├── mesh.xdmf                   Mesh geometry and boundary tags
├── mesh.h5                     HDF5 data store for the mesh
├── dynamic_log.csv             Time-series of total, elastic, dome, and base energies
└── parameters.txt              Plain-text record of all simulation parameters used in this run
```

To visualise the director field in ParaView, open `simulation_n.xdmf` and apply a **Glyph** filter with the `Director` field.

## Support

For technical questions regarding this implementation, please refer to the associated publication or contact the corresponding authors.
