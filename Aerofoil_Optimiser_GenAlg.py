import os
import subprocess
import numpy as np
import random
from scipy.interpolate import interp1d
import shutil
import matplotlib.pyplot as plt

# Parameters
POPULATION_SIZE = 20        # Number of airfoils in each generation
NUM_GENERATIONS = 50        # Number of generations to run
MUTATION_RATE = 0.2         # Probability of a gene mutating
CROSSOVER_RATE = 0.8        # Probability of two parents crossing over
ELITISM_COUNT = 2           # Number of airfoils to carry over to the next generation

# XFoil Parameters
REYNOLDS_NUMBER = 50000    # Re for sim
MACH_NUMBER = 0.02          # Mach for sim
N_ITER = 500                # XFoil solver iterations
START_AOA = -2              # Start angle of attack for the sweep
END_AOA = 10               # End angle of attack
AOA_STEP = 0.5              # Angle of attack increment
NCRIT = 9

# Hicks-Henne
NUM_BUMPS = 8               # Number of Hicks Henne bump functions, these will be split with 4 on the bottom and 4 on the top
BUMP_AMPLITUDE_MAX = 0.0075   # Max amplitude of the bumps as a fraction of chord

# starting airfoil chords
SG6043_COORDS = """
    1.000000   .000000
     .998105   .000656
     .992735   .002712
     .984387   .006072
     .973434   .010465
     .960071   .015523
     .944288   .020916
     .925966   .026547
     .905161   .032475
     .882072   .038683
     .856884   .045099
     .829794   .051648
     .801008   .058240
     .770741   .064777
     .739215   .071138
     .706663   .077173
     .673195   .082684
     .638892   .087606
     .603962   .091903
     .568537   .095505
     .532761   .098415
     .496847   .100592
     .460955   .102010
     .425276   .102690
     .389995   .102599
     .355272   .101752
     .321306   .100174
     .288269   .097880
     .256330   .094905
     .225671   .091274
     .196447   .087021
     .168817   .082199
     .142926   .076839
     .118895   .070995
     .096866   .064730
     .076936   .058094
     .059198   .051170
     .043757   .043981
     .030622   .036590
     .019828   .029159
     .011416   .021706
     .005277   .014355
     .001496   .007410
     .000024   .000942
     .000588  -.004677
     .003930  -.008595
     .010772  -.011195
     .020868  -.013164
     .034086  -.014302
     .050504  -.014598
     .070145  -.014215
     .092915  -.013293
     .118637  -.011970
     .147117  -.010343
     .178147  -.008477
     .211498  -.006446
     .246927  -.004303
     .284172  -.002104
     .322960   .000114
     .363008   .002319
     .404014   .004498
     .445705   .006689
     .487849   .008933
     .530290   .011167
     .572862   .013270
     .615252   .015037
     .657041   .016351
     .697849   .017212
     .737347   .017604
     .775201   .017514
     .811085   .016945
     .844678   .015916
     .875671   .014467
     .903773   .012654
     .928712   .010558
     .950241   .008276
     .968141   .005908
     .982146   .003613
     .992088   .001700
     .998026   .000453
     .999999   .000000
"""


class Airfoil:
    """Airfoil, holds it genes and fitness value"""
    def __init__(self, genes):
        self.genes = genes  # list containing the Hicks-Henne amplitudes that were added to optimise the airfoil
        self.fitness = 0.0  # fitness is defines as the L/D max of the airfoil as this is what were optimising

def plot_airfoil(coords, filename, title):
    """Plots the airfoil shape and saves it to a file"""
    plt.figure(figsize=(12, 6))
    plt.plot(coords[:, 0], coords[:, 1], 'b-')
    plt.title(title)
    plt.xlabel("Chord (x/c)")
    plt.ylabel("Thickness (y/c)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(filename)
    plt.close() 

def load_baseline_airfoil(coords_string):
    """Loads starting airfoil coordinates and returns """
    lines = coords_string.strip().split('\n') # splits chord file up into individual chords
    lines = [line for line in lines if line.strip()] # removes whitespace
    coords = np.array([list(map(float, line.split())) for line in lines]) # gets chords
    
    # Find the split point (leading edge, approximately x=0)
    le_index = np.argmin(coords[:, 0])
    
    # Split into upper and lower surfaces based on standard Selig format
    upper_surface = np.flipud(coords[:le_index+1, :]) # Flip to go from LE to TE
    lower_surface = coords[le_index:, :]             # Already goes from LE to TE
    
    # Create interpolation functions for each surface to create plot
    interp_upper = interp1d(upper_surface[:, 0], upper_surface[:, 1], kind='cubic', fill_value="extrapolate")
    interp_lower = interp1d(lower_surface[:, 0], lower_surface[:, 1], kind='cubic', fill_value="extrapolate")

    return interp_upper, interp_lower

def hicks_henne_bump(x, x_loc, t=0.1):
    """Hicks-Henne bump function"""
    return np.sin(np.pi * np.power(x, np.log(0.5) / np.log(x_loc)))**4

def apply_hicks_henne(genes, base_upper, base_lower):
    """Applies Hicks-Henne bumps to the baseline airfoil."""
    num_bumps_per_surface = len(genes) // 2 # divide by two as half on upper surface and half on lower
    x = np.linspace(0.0, 1.0, 100)
    
    bump_locations = np.linspace(0.1, 0.8, num_bumps_per_surface)
    
    y_upper_mod = np.zeros_like(x)
    y_lower_mod = np.zeros_like(x)
    
    for i in range(num_bumps_per_surface):
        y_upper_mod += genes[i] * hicks_henne_bump(x, bump_locations[i]) # calculates the effect of the bump on all points on upper surface

    for i in range(num_bumps_per_surface):
        y_lower_mod += genes[i + num_bumps_per_surface] * hicks_henne_bump(x, bump_locations[i]) # calculates the effect of the bump on all points on lower surface

    new_y_upper = base_upper(x) + y_upper_mod
    new_y_lower = base_lower(x) + y_lower_mod
    
    if np.any(new_y_lower[1:-1] > new_y_upper[1:-1]): # checks that the airfoil upper and lower sufaces do not intersect
        return None # if they intersect the airfoil is flagged by None to be repaired

    upper = np.vstack((np.flipud(x), np.flipud(new_y_upper))).T # Vertically stacks the upper surface but reverses the order to comply with standard ordering
    lower = np.vstack((x, new_y_lower)).T[1:] # Vertically stacks lower surface
    
    final_coords = np.vstack((upper, lower)) # Verticall stacks both together to create new coords
    return final_coords

def run_xfoil(airfoil_coords, generation, individual_idx):
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
        f"VISC {REYNOLDS_NUMBER}\n"
        f"MACH {MACH_NUMBER}\n"
        "TYPE 1\n"
        f"ITER {N_ITER}\n"
        "PACC\n"
        f"{polar_file}\n"
        "\n"
        f"ASEQ {START_AOA} {END_AOA} {AOA_STEP}\n"
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
            print("--- XFOIL STDOUT ---")
            print(e.stdout)
        if hasattr(e, 'stderr'):
            print("--- XFOIL STDERR ---")
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
            print(f"  -> XFoil ran, but polar file '{polar_file}' is missing or empty.")
            max_ld = -100.0

    except Exception as e:
        print(f"Could not parse polar file for {airfoil_name}. Error: {e}")
        max_ld = -100.0
    finally:
        if os.path.exists(dat_file): os.remove(dat_file)
        if os.path.exists(polar_file): os.remove(polar_file)
        if os.path.exists("xfoil_input.in"): os.remove("xfoil_input.in")

    return max_ld

def evaluate_fitness(individual, base_upper, base_lower, generation="eval", idx=0):
    """Calculates fitness (L/D max) for an individual."""
    coords = apply_hicks_henne(individual.genes, base_upper, base_lower) # Creates new airfoil
    
    if coords is None:
        print(f"  -> Invalid geometry for gen_{generation}_ind_{idx}. Penalizing fitness.")
        individual.fitness = -200.0
        return # If airfoil is invalid the penalty ensures that it wont be continued to the next generation

    individual.fitness = run_xfoil(coords, generation, idx) # runs Xfoil sim to get L/D values

def create_initial_population():
    """Creates the starting population by adding randomly generated bump amplitudes."""
    population = []
    population.append(Airfoil([0.0] * NUM_BUMPS))
    
    for _ in range(POPULATION_SIZE - 1):
        genes = [random.uniform(-BUMP_AMPLITUDE_MAX / 5, BUMP_AMPLITUDE_MAX / 5) for _ in range(NUM_BUMPS)]
        population.append(Airfoil(genes))
    return population

def selection(population):
    """Selects parents using tournament selection."""
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, 5) # selects 5 aerofoils from the population to be compared. The rondomness allows diversification in the genepool as the same parent isnt selected every time
        winner = max(tournament, key=lambda ind: ind.fitness) 
        parents.append(winner) # The best airfoil in the tournament passes on its genome as a parent, this allows strong genes to remain in the genepool
    return parents 

def crossover(parent1, parent2):
    """Performs blended crossover"""
    if random.random() < CROSSOVER_RATE: # checks if the parents will actually breed
        alpha = random.uniform(0.3, 0.7) # blend factor, how much influence each parent has on the child
        child1_genes = [alpha * g1 + (1-alpha) * g2 for g1, g2 in zip(parent1.genes, parent2.genes)] # each gene in both parents are blended with one another using the weighting found above to produce a child
        child2_genes = [(1-alpha) * g1 + alpha * g2 for g1, g2 in zip(parent1.genes, parent2.genes)] # same again but the inverse
        return Airfoil(child1_genes), Airfoil(child2_genes)
    return Airfoil(parent1.genes[:]), Airfoil(parent2.genes[:])

def mutate(individual):
    """Mutates an individual's genes by adding a small change"""
    for i in range(len(individual.genes)):
        if random.random() < MUTATION_RATE: # checks if this gene will be mutated
            mutation_value = random.gauss(0, BUMP_AMPLITUDE_MAX / 5) # chooses a mutation value from a normal distribution centred at zero with st.dev of the max amp / 5 to ensure small mutations
            individual.genes[i] += mutation_value # adds the mutation to form a new gene
            individual.genes[i] = max(-BUMP_AMPLITUDE_MAX, min(BUMP_AMPLITUDE_MAX, individual.genes[i])) # inner function ensures that the gene can neber cause a bump bigger than the max, outer function prevents it from going below the negative limit by taking the max of the negative limit and bump. Ensures the value always sits between the two

def repair_genes(genes, base_upper, base_lower):
    """Repairs a set of genes by scaling them down until they produce a valid geometry"""
    for _ in range(10): 
        coords = apply_hicks_henne(genes, base_upper, base_lower)
        if coords is not None: # checks if the check in the hicks hennes function went off or not
            return genes
        genes = [g * 0.9 for g in genes] # if there is an intersection then the genes are scaled down to remove this and passed back through the loop until the intersection is removed
    return genes

def main():
    """Main function to run the genetic algorithm"""
    if not shutil.which("xfoil.exe"):
        print("ERROR: xfoil.exe not found in your system's PATH.")
        print("Please add the XFoil directory to your PATH or place xfoil.exe in this script's directory.")
        return

    print("Loading baseline airfoil SG6043")
    base_upper, base_lower = load_baseline_airfoil(SG6043_COORDS)
    
    print("Performing baseline geometry check")
    baseline_check_coords = apply_hicks_henne([0.0] * NUM_BUMPS, base_upper, base_lower)
    if baseline_check_coords is None:
        print("The baseline airfoil data resulted in an invalid geometry. Exiting.")
        return
    else:
        print("Baseline geometry is valid. Starting optimization.")
    
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating initial population, output files will be saved in '{output_dir}'")
    population = create_initial_population()

    for gen in range(NUM_GENERATIONS):
        print(f"\nGeneration {gen + 1}/{NUM_GENERATIONS}")

        for i, individual in enumerate(population):
            print(f"Evaluating individual {i+1}/{POPULATION_SIZE}...")
            evaluate_fitness(individual, base_upper, base_lower, gen, i)
            print(f"  -> Fitness (L/D max): {individual.fitness:.4f}")

        population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_of_gen = population[0]
        print(f"\nBest Fitness in Generation {gen + 1}: {best_of_gen.fitness:.4f}")
        print(f"Best Genes: {[f'{g:.6f}' for g in best_of_gen.genes]}")
        
        best_coords = apply_hicks_henne(best_of_gen.genes, base_upper, base_lower)
        
        if best_coords is not None:
            dat_filename = os.path.join(output_dir, f"best_airfoil_gen_{gen+1}.dat")
            np.savetxt(dat_filename, best_coords, fmt='%8.6f', header=f"Best airfoil from generation {gen+1}. L/D_max = {best_of_gen.fitness:.4f}")
            
            plot_filename = os.path.join(output_dir, f"best_airfoil_gen_{gen+1}.png")
            plot_title = f"Best Airfoil Gen {gen+1} | L/D max: {best_of_gen.fitness:.2f}"
            plot_airfoil(best_coords, plot_filename, plot_title)
        else:
            print(f"Best individual of Gen {gen+1} has invalid geometry. No file saved.")

        next_generation = []
        if ELITISM_COUNT > 0:
            next_generation.extend(population[:ELITISM_COUNT])

        parents = selection(population)
        
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            
            mutate(child1)
            mutate(child2)
            
            child1.genes = repair_genes(child1.genes, base_upper, base_lower)
            child2.genes = repair_genes(child2.genes, base_upper, base_lower)
            
            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)
        
        population = next_generation

    print("\nOptimisation Finished")
    final_best = max(population, key=lambda ind: ind.fitness)
    evaluate_fitness(final_best, base_upper, base_lower, "final", 0)

    print(f"Overall Best Fitness (L/D max): {final_best.fitness:.4f}")
    print(f"Overall Best Genes: {final_best.genes}")
    final_coords = apply_hicks_henne(final_best.genes, base_upper, base_lower)
    
    if final_coords is not None:
        final_dat_filename = os.path.join(output_dir, "optimised_airfoil_final.dat")
        np.savetxt(final_dat_filename, final_coords, fmt='%8.6f', header=f"Optimised airfoil. L/D_max = {final_best.fitness:.4f}")
        print(f"\nFinal optimized airfoil saved to '{final_dat_filename}'")

        final_plot_filename = os.path.join(output_dir, "optimised_airfoil_final.png")
        final_plot_title = f"Final Optimised Airfoil | L/D max: {final_best.fitness:.2f}"
        plot_airfoil(final_coords, final_plot_filename, final_plot_title)
        print(f"Plot of final airfoil saved to '{final_plot_filename}'")
    else:
        print("\nFinal best airfoil had invalid geometry. No file saved.")


if __name__ == "__main__":
    main()
