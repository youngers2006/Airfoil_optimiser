import os
import subprocess
import numpy as np
import random
from scipy.interpolate import interp1d
import shutil
import matplotlib.pyplot as plt
from Xfoil_operations import run_xfoil, Param_container
from GenAlg_operations import create_initial_population, repair_genes, evaluate_fitness, mutate, selection, crossover
from Hicks_Hennes_operations import apply_hicks_henne, hicks_henne_bump
from Aerofoil_operations import Airfoil, plot_airfoil, load_baseline_airfoil

def main():
    """Main function to run the genetic algorithm"""

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
    AEROFOIL_COORDS = """
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

    Parameters = Param_container(
        REYNOLDS_NUMBER,
        MACH_NUMBER,
        N_ITER,
        START_AOA,
        END_AOA,
        AOA_STEP,
        NCRIT,
        NUM_BUMPS,
        BUMP_AMPLITUDE_MAX,
        AEROFOIL_COORDS,
        POPULATION_SIZE,
        NUM_GENERATIONS,
        MUTATION_RATE,
        CROSSOVER_RATE,
        ELITISM_COUNT
    )

    if not shutil.which("xfoil.exe"):
        print("error: xfoil.exe not found in your system's PATH.")
        print("Please add the XFoil directory to your PATH or place xfoil.exe in this script's directory.")
        return

    print("Loading baseline airfoil")
    base_upper, base_lower = load_baseline_airfoil(AEROFOIL_COORDS)
    
    print("Performing baseline geometry check")
    baseline_check_coords = apply_hicks_henne([0.0] * NUM_BUMPS, base_upper, base_lower)
    if baseline_check_coords is None:
        print("The baseline airfoil data resulted in an invalid geometry. Exiting.")
        return
    else:
        print("Baseline geometry is valid. Starting optimization.")
    
    output_dir = "optimisation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating initial population, output files will be saved in '{output_dir}'")
    population = create_initial_population(Parameters)

    for gen in range(NUM_GENERATIONS):
        print(f"\nGeneration {gen + 1}/{NUM_GENERATIONS}")

        for i, individual in enumerate(population):
            print(f"Evaluating individual {i+1}/{POPULATION_SIZE}...")
            evaluate_fitness(individual, base_upper, base_lower, Parameters, gen, i)
            print(f"Fitness (L/D max): {individual.fitness:.4f}")

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
            next_generation.extend(population[:Parameters.elite_count])

        parents = selection(population)
        
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2, Parameters)
            
            mutate(child1, Parameters)
            mutate(child2, Parameters)
            
            child1.genes = repair_genes(child1.genes, base_upper, base_lower)
            child2.genes = repair_genes(child2.genes, base_upper, base_lower)
            
            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)
        
        population = next_generation

    print("\nOptimisation Finished")
    final_best = max(population, key=lambda ind: ind.fitness)
    evaluate_fitness(final_best, base_upper, base_lower, Parameters, "final", 0)

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