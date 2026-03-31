"""
### main_JPOM.py

Entry point for the Polarized Optical Microscopy (POM) image generation pipeline.
Responsibilities:
  - Instantiate POMParameters (all user-input settings live there)
  - Hand off to run_pom_pipeline() in src_func/pom_generator.py
  - Report total wall-clock time and catch any fatal exceptions

Usage:
    python main_JPOM.py
"""

import time
import traceback

from params_JPOM import POMParameters
from src_func.pom_generator import run_pom_pipeline


def main():
    """
    Top-level driver.

    Creates a POMParameters instance, runs the pipeline, and reports
    total elapsed time.  Returns 0 on success, 1 on any unhandled error
    so the exit code can be checked by shell scripts.
    """
    start_time = time.time()

    print("\nJones Matrix POM Pipeline (v2.2) ---\n")

    # Build the parameter object (edit params_JPOM.py to change settings)
    params = POMParameters()

    try:
        # Run the full pipeline; returns the output directory path
        output_dir = run_pom_pipeline(params)
    except Exception:
        # Print the full traceback so the user can see where it failed
        print("\n" + "=" * 50 + "\nFATAL ERROR:\n" + "=" * 50)
        traceback.print_exc()
        return 1   # Non-zero exit → failure

    elapsed = time.time() - start_time
    print(f"\nTotal pipeline time: {elapsed:.1f}s\n")
    return 0   # Zero exit → success


if __name__ == "__main__":
    exit(main())
