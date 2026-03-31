"""
### params_JPOM.py

Edit this file to change the simulation directory, optical constants,
image resolution, polarizer angles, etc.

All physical lengths are stored in SI units internally, but the
user sets them in the natural units shown in the comments.
"""

import numpy as np


class POMParameters:
    """
    Container for every tunable setting in the POM pipeline.

    Attributes are grouped by purpose; see inline comments for units and meaning.
    """

    def __init__(self):
        ## INPUT .h5 MESH DATA PATH
        # Absolute path to the FEniCSx/DOLFINx simulation directory.
        # The pipeline expects a file called 'simulation_P.h5' inside this folder.
        self.simulation_dir = ("/home/zk/Documents/PyScripts/SmZA_Cooling/")

        ## OPTICAL PARAMETERS
        # Extraordinary refractive index (n_e) of the liquid crystal.
        # Ordinary refractive index (n_o) of the liquid crystal.
        # Birefringence Δn = N_E − N_O.
        self.N_E = 1.68
        self.N_O = 1.5

        ## IMAGE RESOLUTION
        # Number of pixels along each axis of the square output image.
        # Higher values give more detail but increase runtime quadratically.
        self.resolution_xy = 400

        ## POLARIZER ROTATION ANGLES
        # Iterable of angles (degrees) at which to generate POM frames.
        # Each angle rotates both polarizer and analyzer together (crossed pair).
        self.angles = range(0, 50, 15)

        ## Z-SAMPLING (column discretisation)
        # Number of equally-spaced sample points along each pixel column (z-axis).
        # More samples → smoother Jones matrix product; 40–80 is usually sufficient.
        self.n_z_samples = 80

        ## DEBUG PLOTS
        # Set to 0 to skip entirely (speeds up the pipeline significantly).
        # Each layer plot is a 6-panel figure saved to debug_plots/.
        self.n_debug_layers = 0

        ## DUMMY LAYERS (substrate compensation)
        self.n_dummy_layers_bottom = 0

        ## TIMESTEP SELECTION
        # Which simulation snapshot to load from the HDF5 file.
        #   -1  → last available timestep
        #    n  → n-th timestep (0-indexed)
        self.timestep = -1

        ## CONTACT ANGLE OVERRIDE (UNIT : Degrees)
        # If set to a float (degrees), the droplet height is computed from this
        # contact angle via:  h = R * tan(θ / 2)
        # If None, the height is taken directly from the z-extent of the mesh.
        self.contact_angle_deg = None

        ## INTENSITY NORMALISATION
        # If True, each POM frame is rescaled so the brightest pixel = 1.0.
        # Set to False to preserve inter-frame intensity ratios.
        self.normalize_intensity = True
