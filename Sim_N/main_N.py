"""
main_N.py

Entry point for the Nematic Liquid Crystal Ginzburg-Landau dynamics simulation.

Usage
-----
    mpirun -n 4 python main_N.py      # parallel run (recommended)
    python main_N.py                  # serial run
"""

import time
import traceback

from mpi4py import MPI
from params_N import SimulationParameters
from src.solver import run_simulation


def main():
    rank = MPI.COMM_WORLD.rank
    start_time = time.time()

    sim_params = SimulationParameters()

    try:
        output_dir = run_simulation(sim_params)

    except Exception as e:
        if rank == 0:
            print(f"\n" + "=" * 50 + "\nFATAL ERROR:\n" + "=" * 50)
            traceback.print_exc()
        return 1

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"\nTotal simulation time: {elapsed:.1f} s")
        print(f"Output written to:     {output_dir}/")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
    
