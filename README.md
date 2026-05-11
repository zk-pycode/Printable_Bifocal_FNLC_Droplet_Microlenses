## Printable Bifocal Microlenses from Ferroelectric Nematic Liquid Crystal Droplets
<a href="https://doi.org/10.5281/zenodo.20127570"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20127570.svg" alt="DOI"></a>
<a href="https://doi.org/10.5281/zenodo.20127570"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20127570.svg" alt="DOI"></a>
### Project Description

This project models the equilibrium director configurations of liquid crystal (LC) droplets across three material phases - **Nematic (N)**, **Intermediate-Phase (Nx or SmZa)**, and **Ferroelectric Nematic (FN)** - each implemented as a separate finite-element simulation module.

In all three phases the director field is obtained by minimizing the Landau-de Gennes (LdG) free energy functional via Ginzburg-Landau relaxation on a spherical-cap tetrahedral mesh, subject to surface anchoring conditions at the dome and polyamide substrate. The three phases differ in the structure of the free energy and the governing equations:
```
Project_Structure/
├── Sim_N/              LdG free energy, frank-olseen elastic anchoring (Q-tensor formulation)
├── Sim_NX/             LdG + covariant smectic order parameter, coupled Poisson solve for polar terms
├── Sim_NF/             LdG + spontaneous polarization, coupled Poisson solve for bound charges
└── JM_POM/             Jones matrix propagation, Michel-Levy colors
```
The simulated steady-state director textures from each phase are converted to synthetic polarized optical microscopy (POM) images using the Jones matrix calculus pipeline. Layer-by-layer propagation of the Jones electric-field vector through the stratified LC stack, integrated over the visible spectrum, produces Michel-Levy colored POM frames that can be directly compared with experimental micrographs for quantitative validation.

---

### References

```
Siddiquee, Z., & Jákli, A. (2026). FNLC Bifocal Microlenses (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20127570

TBA
```
