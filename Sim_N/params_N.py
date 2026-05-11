"""
### params_N.py

Edit this file to change the droplet geometry, elastic constants,
anchoring strengths, time-stepping, and output settings.

All physical lengths and energies are stored in SI units internally,
but the comments note the natural units for each quantity.
"""

import numpy as np


class SimulationParameters:
    """
    Container for every tunable setting in the Nematic LC simulation.

    Attributes are grouped by purpose; see inline comments for units and meaning.
    """

    def __init__(self):

        ## SCALAR ORDER PARAMETER
        # Equilibrium value of the nematic order parameter S.
        # Enters the Frank→LdG elastic mapping and anchoring energy conversions.
        self.S = 0.5

        ## FRANK ELASTIC CONSTANTS  (Unit: N = J/m, typical range: 1–20 pN)
        # Splay (K1), twist (K2), and bend (K3) elastic constants.
        self.K1 = 0.4e-12               # N - splay
        self.K2 = 1.0e-12               # N - twist
        self.K3 = 2.0e-12               # N - bend

        ## LdG ELASTIC CONSTANTS  (Unit: N, derived from Frank constants)
        # Mapping from one-constant Frank theory to Landau–de Gennes:
        #   K1 = 2S²(L1 + L2 - L3)
        #   K2 = 2S²·L1
        #   K3 = 2S²(L1 + L2 + L3)
        S2 = self.S ** 2
        self.L1 = self.K2 / (2.0 * S2)
        self.L2 = (self.K1 - 2.0 * self.K2 + self.K3) / (4.0 * S2)
        self.L3 = (self.K3 - self.K1) / (4.0 * S2)

        ## SURFACE ANCHORING  (Unit: J/m²)
        # C_surface   - planar degenerate anchoring on BOTH dome and base:
        #                   dome: F = C/2 · (ν^T Q ν + S/3)²
        #                   base: F = C/2 · (ẑ^T Q ẑ + S/3)²  (tilt suppression)
        #               physical W = C·S²;  extrapolation length ξ = K/W
        # C_polyamide - Rapini-Papoular anchoring toward the rubbing direction at base:
        #                   F = C/2 · |Q − Q_s|²;  physical W = C·S²
        # Elastic extrapolation length: ξ = K / W  (should satisfy ξ ≪ R_droplet).
        self.C_surface   = 1.0e-6     # J/m² - planar degenerate (dome + base tilt)
        self.C_polyamide = 2.0e-8     # J/m² - base Rapini-Papoular (rubbing direction)

        ## BASE RUBBING DIRECTION
        # In-plane unit vector (xy-plane) defining the easy axis at the substrate.
        # The preferred Q-tensor is Q_s = S·(d̂⊗d̂ − I/3).
        self.D_rub = np.array([1.0, 0.0])   # along +x

        ## DROPLET GEOMETRY  (Unit: m for lengths, degrees for angles)
        self.base_radius       = 127e-6     # m       - droplet base radius
        self.contact_angle_deg = 20         # degrees - static contact angle
        self.contact_angle_rad = np.deg2rad(self.contact_angle_deg)

        ## MESH PARAMETERS
        # mesh_size_factor scales the gmsh element size relative to the base radius.
        # min_bulk_layers controls the minimum vertical resolution through the
        # droplet height; increase to better resolve twist or thin surface layers.
        self.mesh_size_factor  = 0.1        # mesh size = factor × base_radius
        self.mesh_size         = self.base_radius * self.mesh_size_factor
        self.min_bulk_layers   = 3.0        # minimum tetrahedral layers in bulk

        ## DYNAMICS  (Unit: Pa·s for viscosity, s for time)
        # gamma_viscosity - rotational viscosity γ₁.
        # dt              - time step; elastic relaxation τ ≈ γ·R²/K ~ 700 s,
        #                   so dt ≪ 7 s keeps the BDF scheme well within stability.
        self.gamma_viscosity = 0.06         # Pa·s - rotational viscosity
        self.dt              = 0.5          # s     - time step
        self.num_steps       = 10000         # total steps → t_final = 2500 s ≈ 3.5τ

        ## SOLVER
# SNES (Newton) convergence tolerances applied at each time step.
        self.newton_max_iter = 100          # maximum Newton iterations per step
        self.newton_tol      = 1e-7         # residual convergence tolerance

        ## INITIAL CONDITION
        # Choose 'random', 'radial', or 'from_file'.
        # 'from_file' requires ic_xdmf_file to point to a previous simulation output.
        self.initial_condition_type = 'random'
        self.ic_xdmf_file = ('...')

        ## OUTPUT
        # XDMF snapshots are written every output_every_n_steps time steps.
        self.output_dir           = "Nematic_0001"
        self.output_every_n_steps = 5       # save XDMF every N steps
