## Printable Bifocal Microlenses from Ferroelectric Nematic Liquid Crystal Droplets

## Project Description

This project models the equilibrium director configurations of liquid crystal (LC) droplets across three material phases — **nematic (N)**, **smectic-Z_A (SmZ_A)**, and **ferroelectric nematic (FN)** — each implemented as a separate finite-element simulation module.

In all three phases the director field is obtained by minimizing the Landau-de Gennes (LdG) free energy functional via Ginzburg-Landau relaxation on a spherical-cap tetrahedral mesh, subject to surface anchoring conditions at the dome and polyamide substrate. The three phases differ in the structure of the free energy and the governing equations:
```
BetteryLabs/
├── Sim_N/              Nematic Q-tensor simulation (Frank-Oseen, BDF1/BDF2)
├── Sim_NX/             Smectic-Z_A simulation (Oda & Fukuda 2025, covariant gradient)
├── Sim_NF/             Ferroelectric nematic simulation (polar + Poisson coupling)
└── JM_POM/             POM image generator (Jones matrix, Michel-Levy colors)
```
The simulated steady-state director textures from each phase are converted to synthetic polarized optical microscopy (POM) images using the **JonesMatrix_POM** Jones matrix calculus pipeline. Layer-by-layer propagation of the Jones electric-field vector through the stratified LC stack, integrated over the visible spectrum, produces Michel-Levy colored POM frames that can be directly compared with experimental micrographs for quantitative validation.

---

## References

### Journal Article
```
TBA
```

### Software
```
Z. Siddiquee, M. Talwar, J. Selinger, and A. Jákli,
*Printable Bifocal FNLC Droplet Microlenses* (Version 2.0.4), 2026.
Zenodo. DOI: [10.0000/zenodo.----](https://doi.org/10.0000/zenodo.----)
URL: https://github.com/zk-pycode/Printable_Bifocal_FNLC_Droplet_Microlenses
```
