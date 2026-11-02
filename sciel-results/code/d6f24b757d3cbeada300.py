"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm with multi-objective fitness.
    Fitness penalizes both the number of blocks and the number of uncovered t-subsets.
    """
    # Parameters
    population_size = 200
    max_generations = 150
    crossover_rate = 0.8
    mutation_rate = 0.15
    elite_size = 20
    tournament_size = 5
    time_limit = 100  # seconds, leaving buffer for evaluation
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each possible block (k-subset), which t-subsets it covers
    # This is expensive, so we'll compute coverage on the fly but cache results
    # Actually, let's precompute the coverage mask for each block type
    # There are C(19,8) = 75582 possible blocks, each covers C(8,5) = 56 t-subsets
    
    # Instead of precomputing all, we'll use a bitset approach
    # Represent coverage as a numpy boolean array of length num_tsubsets
    
    def block_coverage(block):
        """Return a boolean array indicating which t-subsets are covered by this block."""
        cov = np.zeros(num_tsubsets, dtype=bool)
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            cov[idx] = True
        return cov
    
    def total_coverage(blocks):
        """Return a boolean array of all t-subsets covered by the given blocks."""
        cov = np.zeros(num_tsubsets, dtype=bool)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                cov[idx] = True
        return cov
    
    def fitness(blocks):
        """
        Multi-objective fitness: lower is better.
        Primary: number of uncovered t-subsets
        Secondary: number of blocks (tie-breaker)
        """
        cov = total_coverage(blocks)
        uncovered_count = num_tsubsets - np.sum(cov)
        return uncovered_count * 10000 + len(blocks)
    
    def random_block():
        """Generate a random k-subset of range(v)."""
        return sorted(random.sample(range(v), k))
    
    def random_individual():
        """Generate a random individual (list of blocks)."""
        # Start with a reasonable number of blocks
        # Lower bound is 264, best known is 335, so try around 300-400
        num_blocks = random.randint(280, 400)
        blocks = set()
        attempts = 0
        while len(blocks) < num_blocks and attempts < num_blocks * 5:
            b = tuple(random_block())
            blocks.add(b)
            attempts += 1
        return [list(b) for b in blocks]
    
    def greedy_individual():
        """Generate an individual using greedy covering as a starting point."""
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered:
            # Pick the first uncovered t-subset
            target = next(iter(uncovered))
            # Build a block containing this target
            block_set = set(target)
            while len(block_set) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for ts in itertools.combinations(block_set, t - 1):
                        new_ts = tuple(sorted(ts + (x,)))
                        if new_ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block_set.add(best_point)
            block = sorted(block_set)
            blocks.append(block)
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(tuple(ts))
        return blocks
    
    # Initialize population
    population = []
    
    # Add some greedy individuals for quality
    for _ in range(30):
        population.append(greedy_individual())
    
    # Add random individuals for diversity
    while len(population) < population_size:
        population.append(random_individual())
    
    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]
    
    # Sort by fitness
    for _ in range(10):  # A few generations to let GA work
        pass
    
    best_individual = min(population, key=lambda ind: fitness(ind))
    best_fitness = fitness(best_individual)
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate current population if not already done
        if gen == 0:
            fitnesses = [fitness(ind) for ind in population]
        
        # Sort population by fitness
        indexed_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Update best
        if indexed_pop[0][0] < best_fitness:
            best_fitness = indexed_pop[0][0]
            best_individual = indexed_pop[0][1]
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        for i in range(elite_size):
            new_population.append(indexed_pop[i][1])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(range(len(indexed_pop)), min(tournament_size, len(indexed_pop)))
                parent_idx = min(candidates, key=lambda i: indexed_pop[i][0])
                parents.append(indexed_pop[parent_idx][1])
            
            # Crossover
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = list(parents[0])
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [fitness(ind) for ind in new_population]
        population = new_population
    
    return best_individual


def crossover(parent1, parent2):
    """
    Crossover operator: combine blocks from two parents.
    We use a uniform crossover where each block is taken from one of the parents with some probability.
    """
    # Merge unique blocks from both parents
    blocks1 = set(tuple(b) for b in parent1)
    blocks2 = set(tuple(b) for b in parent2)
    
    # With some probability, take a block from parent1, else from parent2
    child = []
    seen = set()
    
    # First, include some blocks from the better parent (assume parent1 is better)
    for b in parent1:
        bt = tuple(b)
        if bt not in seen:
            child.append(b)
            seen.add(bt)
    
    # Add some blocks from parent2 to increase diversity
    for b in parent2:
        bt = tuple(b)
        if bt not in seen:
            child.append(b)
            seen.add(bt)
    
    # Shuffle
    random.shuffle(child)
    
    return child


def mutate(individual):
    """
    Mutation operator: replace some blocks with new random blocks.
    """
    n = len(individual)
    num_mutations = max(1, n // 10)
    
    for _ in range(num_mutations):
        idx = random.randint(0, n - 1)
        # Replace with a random block
        new_block = sorted(random.sample(range(19), 8))
        individual[idx] = new_block
    
    return individual
# EVOLVE-BLOCK-END
