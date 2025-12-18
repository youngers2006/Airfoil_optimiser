import random
from Hicks_Hennes_operations import apply_hicks_henne
from Xfoil_operations import run_xfoil
from Aerofoil_operations import Airfoil

def evaluate_fitness(individual, base_upper, base_lower, Parameters, generation="eval", idx=0):
    """Calculates fitness (L/D max) for an individual."""
    coords = apply_hicks_henne(individual.genes, base_upper, base_lower) # Creates new airfoil
    
    if coords is None:
        print(f"  -> Invalid geometry for gen_{generation}_ind_{idx}. Penalizing fitness.")
        individual.fitness = -200.0
        return # If airfoil is invalid the penalty ensures that it wont be continued to the next generation

    individual.fitness = run_xfoil(coords, generation, idx, Parameters) # runs Xfoil sim to get L/D values

def create_initial_population(Parameters):
    """Creates the starting population by adding randomly generated bump amplitudes."""
    population = []
    population.append(Airfoil([0.0] * Parameters.Num_HH_bumps))
    
    for _ in range(Parameters.pop_size - 1):
        genes = [random.uniform(-Parameters.HH_max_amp / 5, Parameters.HH_max_amp / 5) for _ in range(Parameters.Num_HH_bumps)]
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

def crossover(parent1, parent2, Parameters):
    """Performs blended crossover"""
    if random.random() < Parameters.cross_rate: # checks if the parents will actually breed
        alpha = random.uniform(0.3, 0.7) # blend factor, how much influence each parent has on the child
        child1_genes = [alpha * g1 + (1-alpha) * g2 for g1, g2 in zip(parent1.genes, parent2.genes)] # each gene in both parents are blended with one another using the weighting found above to produce a child
        child2_genes = [(1-alpha) * g1 + alpha * g2 for g1, g2 in zip(parent1.genes, parent2.genes)] # same again but the inverse
        return Airfoil(child1_genes), Airfoil(child2_genes)
    return Airfoil(parent1.genes[:]), Airfoil(parent2.genes[:])

def mutate(individual, Parameters):
    """Mutates an individual's genes by adding a small change"""
    for i in range(len(individual.genes)):
        if random.random() < Parameters.mut_rate: # checks if this gene will be mutated
            mutation_value = random.gauss(0, Parameters.HH_max_amp / 5) # chooses a mutation value from a normal distribution centred at zero with st.dev of the max amp / 5 to ensure small mutations
            individual.genes[i] += mutation_value # adds the mutation to form a new gene
            individual.genes[i] = max(-Parameters.HH_max_amp, min(Parameters.HH_max_amp, individual.genes[i])) # inner function ensures that the gene can neber cause a bump bigger than the max, outer function prevents it from going below the negative limit by taking the max of the negative limit and bump. Ensures the value always sits between the two

def repair_genes(genes, base_upper, base_lower):
    """Repairs a set of genes by scaling them down until they produce a valid geometry"""
    for _ in range(10): 
        coords = apply_hicks_henne(genes, base_upper, base_lower)
        if coords is not None: # checks if the check in the hicks hennes function went off or not
            return genes
        genes = [g * 0.9 for g in genes] # if there is an intersection then the genes are scaled down to remove this and passed back through the loop until the intersection is removed
    return genes