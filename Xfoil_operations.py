import os
import subprocess
import numpy as np

POPULATION_SIZE = 20        # Number of airfoils in each generation
NUM_GENERATIONS = 50        # Number of generations to run
MUTATION_RATE = 0.2         # Probability of a gene mutating
CROSSOVER_RATE = 0.8        # Probability of two parents crossing over
ELITISM_COUNT = 2  

class Param_container:
    """Container to store all parameters."""
    def __init__(
            self, 
            Re, 
            M, 
            N, 
            AoA0, 
            AoAf, 
            AoA_step, 
            N_crit, 
            Num_HH_bumps, 
            HH_max_amp, 
            initial_coords,
            pop_size,
            num_gens,
            mut_rate,
            cross_rate,
            elite_count
        ):
        self.Re = Re
        self.M = M
        self.N = N
        self.AoA0 = AoA0
        self.AoAf = AoAf
        self.AoA_step = AoA_step
        self.N_crit = N_crit
        self.Num_HH_bumps = Num_HH_bumps
        self.HH_max_amp = HH_max_amp
        self.coords = initial_coords
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.elite_count = elite_count

def run_xfoil(airfoil_coords, generation, individual_idx, Parameters):
    """Runs XFoil and returns the max L/D via a 2d simulation."""
    airfoil_name = f"gen_{generation}_ind_{individual_idx}"
    dat_file = f"{airfoil_name}.dat"
    polar_file = f"{airfoil_name}_polar.txt"
    
    np.savetxt(dat_file, airfoil_coords, fmt='%8.6f')
    xfoil_script = (
        "PLOP\n"
        "G F\n"
        "\n"
        f"LOAD {dat_file}\n"
        "\n"
        "PANE\n"
        "OPER\n"
        f"VISC {Parameters.Re}\n"
        f"MACH {Parameters.M}\n"
        "TYPE 1\n"
        f"ITER {Parameters.N}\n"
        "PACC\n"
        f"{polar_file}\n"
        "\n"
        f"ASEQ {Parameters.AoA0} {Parameters.AoAf} {Parameters.AoA_step}\n"
        "\n"
        "PACC\n"
        "QUIT\n"
    )
    
    with open("xfoil_input.in", "w") as f:
        f.write(xfoil_script)

    try:
        with open("xfoil_input.in", 'r') as f_in:
             subprocess.run(
                ["xfoil.exe"],
                stdin=f_in,
                check=True,
                timeout=60,
                capture_output=True,
                text=True
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"XFoil failed for {airfoil_name}. Error: {e}")
        if hasattr(e, 'stdout'):
            print("XFOIL STDOUT")
            print(e.stdout)
        if hasattr(e, 'stderr'):
            print("XFOIL STDERR")
            print(e.stderr)

        if os.path.exists(dat_file): os.remove(dat_file)
        if os.path.exists(polar_file): os.remove(polar_file)
        return -100.0

    max_ld = -100.0
    try:
        if os.path.exists(polar_file) and os.path.getsize(polar_file) > 0:
            polar_data = np.loadtxt(polar_file, skiprows=12)
            valid_indices = polar_data[:, 2] > 1e-5
            if np.any(valid_indices):
                ld_ratios = polar_data[valid_indices, 1] / polar_data[valid_indices, 2]
                max_ld = np.max(ld_ratios) if ld_ratios.size > 0 else -100.0
        else:
            print(f"XFoil ran, but polar file '{polar_file}' is missing or empty.")
            max_ld = -100.0

    except Exception as e:
        print(f"Could not parse polar file for {airfoil_name}. Error: {e}")
        max_ld = -100.0
    finally:
        if os.path.exists(dat_file): os.remove(dat_file)
        if os.path.exists(polar_file): os.remove(polar_file)
        if os.path.exists("xfoil_input.in"): os.remove("xfoil_input.in")

    return max_ld