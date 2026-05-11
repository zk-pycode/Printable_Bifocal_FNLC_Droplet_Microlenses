# Printable Bifocal Microlenses from Ferroelectric Nematic Liquid Crystal Droplets

## Project Description

This project models the equilibrium director configurations of liquid crystal (LC) droplets
across three material phases — **nematic (N)**, **smectic-Z_A (SmZ_A)**, and **ferroelectric
nematic (FN)** — each implemented as a separate finite-element simulation module.

In all three phases the director field is obtained by minimizing the Landau-de Gennes (LdG)
free energy functional via Q-tensor Ginzburg-Landau relaxation on a spherical-cap tetrahedral
mesh, subject to surface anchoring conditions at the dome and polyamide substrate. The three
phases differ in the structure of the free energy and the governing equations:

- **Nematic (N_LC2)** — purely elastic Frank-Oseen free energy with three elastic constants
  (splay K1, twist K2, bend K3). The director field is evolved by a single Q-tensor PDE with
  planar degenerate anchoring at the dome and Rapini-Papoular anchoring at the substrate.

- **Smectic-Z_A (SmZA / NXLC)** — adds a complex smectic order parameter psi = psi_re + i*psi_im
  governed by a covariant gradient free energy (Oda & Fukuda 2025 formulation). The director n
  and psi are coupled and relaxed simultaneously. The antiferroelectric SmZ_A phase introduces
  spontaneous polarization, requiring **Poisson's equation** to be solved concurrently with the
  director relaxation in order to account for the bound charge distribution arising from the
  polar free energy terms.

- **Ferroelectric Nematic (FNLC)** — the fully polar phase with spontaneous bulk polarization
  P_b = sqrt(-A/B). The free energy includes flexoelectric coupling and a polarization gradient
  term (KP). Poisson's equation is again solved concurrently, self-consistently coupling the
  electric potential to the director and polarization fields throughout the droplet volume.

The simulated steady-state director textures from each phase are converted to synthetic
polarized optical microscopy (POM) images using the **JonesMatrix_POM** Jones matrix calculus
pipeline. Layer-by-layer propagation of the Jones electric-field vector through the stratified
LC stack, integrated over the visible spectrum, produces Michel-Levy colored POM frames that
can be directly compared with experimental micrographs for quantitative validation.

---

## Project Structure

```
BetteryLabs/
├── N_LC2/              Nematic Q-tensor simulation (Frank-Oseen, BDF1/BDF2)
├── SmZA/               Smectic-Z_A simulation (Oda & Fukuda 2025, covariant gradient)
├── NXLC/               SmZa polar smectic simulation (antiferroelectric + Poisson coupling)
├── FNLC/               Ferroelectric nematic simulation (polar + Poisson coupling)
└── JonesMatrix_POM/    POM image generator (Jones matrix, Michel-Levy colors)
```
---

## References

### Journal Article
```
M. Talwar, Z. Siddiquee, and A. Jákli,
"Printable Bifocal Microlenses from Ferroelectric Nematic Liquid Crystal Droplets,"
*ACS Appl. Mater. & Interfaces*, May 2026.
DOI: [10.----/acsami.----](https://doi.org/10.----/acsami.----)
```

### Software
```
Z. Siddiquee, M. Talwar, J. Selinger, and A. Jákli,
*Printable Bifocal FNLC Droplet Microlenses* (Version 2.0.4), 2026.
Zenodo. DOI: [10.0000/zenodo.----](https://doi.org/10.0000/zenodo.----)
URL: https://github.com/zk-pycode/Printable_Bifocal_FNLC_Droplet_Microlenses
```
