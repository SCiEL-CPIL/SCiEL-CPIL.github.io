"""Genetic algorithm approach for covering design C(19,8,5) using bitset integers."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm.
    Uses Python integers as bitsets for fast coverage tracking.
    """
    # Parameters
    population_size = 200
    max_generations = 300
    crossover_rate = 0.8
    mutation_rate = 0.4
    elite_size = 20
    tournament_size = 5
    time_limit = 110  # seconds, leaving buffer for evaluation
    
    # Precompute all t-subsets and map to bit indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_total = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage mask for each block as a Python integer
    # Bit i is set if t-subset i is covered by the block
    full_mask = (1 << num_tsubsets) - 1
    block_masks = [0] * num_blocks_total
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Helper to get mask from block tuple
    def get_mask(block):
        return block_masks[block_to_idx[tuple(block)]]
        
    def compute_coverage(blocks):
        """Compute total coverage as a single integer bitset."""
        cov = 0
        for block in blocks:
            cov |= get_mask(block)
        return cov

    def fitness(blocks):
        """
        Multi-objective fitness: lower is better.
        Primary: number of uncovered t-subsets
        Secondary: number of blocks
        """
        cov = compute_coverage(blocks)
        uncovered_count = (full_mask & ~cov).bit_count()
        # We want to minimize uncovered, then minimize size
        # If uncovered is 0, we just minimize size
        # If uncovered > 0, we heavily penalize it
        return uncovered_count * 10000 + len(blocks)

    def random_block():
        return sorted(random.sample(range(v), k))
        
    def random_individual():
        """Generate a random individual."""
        # Start with a size that is likely to cover everything
        # C(19,8,5) lower bound is 264. Best known 335.
        # Random 300 blocks usually cover a lot.
        num_blocks = random.randint(300, 350)
        blocks = set()
        for _ in range(num_blocks * 2):
            if len(blocks) >= num_blocks:
                break
            b = tuple(random_block())
            blocks.add(b)
        return [list(b) for b in blocks]

    def greedy_individual():
        """
        Fast greedy: pick blocks that cover the most *new* subsets.
        To speed up, we don't check all 75k blocks every time if we can avoid it,
        but with bitsets, checking 75k blocks is (75k * 1 bitwise op).
        75k ops is fast in Python? ~0.01s?
        300 steps * 0.01s = 3s per individual.
        With 50 individuals, that's 150s. Too slow.
        
        Let's use a "Random Greedy" or just rely on Random Init + GA.
        Or optimize Greedy:
        Maintain a list of "candidate" blocks?
        Actually, let's just use a simple heuristic:
        Pick a random uncovered t-subset. Pick a random block containing it.
        This is very fast.
        """
        uncovered = full_mask
        blocks = []
        # Map t-subset index to list of blocks containing it?
        # That's expensive to build (11628 * ~1000 blocks).
        
        # Alternative:
        # Just do a few rounds of random blocks, then local search.
        # Let's stick to random init for speed, and let GA do the work.
        # But random 300 blocks might have many gaps.
        
        # Let's try a "Greedy by Random Target":
        # 1. Pick random uncovered t-subset.
        # 2. Pick random block covering it.
        # 3. Add block.
        # This is O(1) per step if we can find a block covering a t-subset fast.
        # Precompute: for each t-subset, list of block indices covering it.
        
        # Let's build this map once.
        # tsubset_to_blocks = {ts: [block_indices]}
        
        return None # Placeholder, we will implement a better init below

    # Precompute t-subset to blocks map for fast greedy
    tsubset_to_blocks = [[] for _ in range(num_tsubsets)]
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            tsubset_to_blocks[idx].append(i)

    def fast_greedy_individual():
        """
        Greedy: repeatedly pick a random uncovered t-subset,
        then pick a random block covering it.
        """
        uncovered_indices = list(range(num_tsubsets))
        random.shuffle(uncovered_indices)
        
        # We need to know which are uncovered.
        # Using a set of indices is fast.
        uncovered_set = set(range(num_tsubsets))
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_set))
            
            # Pick a random block covering this t-subset
            # tsubset_to_blocks[target_idx] contains indices of blocks
            candidate_blocks = tsubset_to_blocks[target_idx]
            if not candidate_blocks:
                break # Should not happen
            
            block_idx = random.choice(candidate_blocks)
            block = all_blocks[block_idx]
            blocks.append(block)
            
            # Update uncovered set
            mask = block_masks[block_idx]
            # Remove all covered t-subsets from uncovered_set
            # Iterating over all 11628 is slow.
            # Iterate over set bits in mask?
            # Or just check if bit is set in uncovered_set?
            # Better: 
            # We can just check the specific t-subsets of the block.
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered_set.discard(idx)
                
        return [list(b) for b in blocks]

    # Initialize population
    population = []
    
    # Mix of random and greedy
    for _ in range(population_size // 2):
        population.append(random_individual())
    for _ in range(population_size // 2):
        population.append(fast_greedy_individual())
        
    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]
    
    # Find initial best
    best_idx = int(np.argmin(fitnesses))
    best_individual = population[best_idx][:]
    best_fitness = fitnesses[best_idx]
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort population by fitness
        indexed_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Update best
        if indexed_pop[0][0] < best_fitness:
            best_fitness = indexed_pop[0][0]
            best_individual = indexed_pop[0][1][:]
            
        # If we found a valid covering, we can try to reduce size further
        # by local search on the best individual.
        if best_fitness < 10000: # Uncovered is 0
            # Try to remove blocks
            improved = True
            while improved:
                improved = False
                b = best_individual[:]
                for i in range(len(b)):
                    # Try removing b[i]
                    temp = b[:i] + b[i+1:]
                    if fitness(temp) < fitness(b):
                        b = temp
                        best_individual = b
                        best_fitness = fitness(b)
                        improved = True
                        break # Restart loop as list changed
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(indexed_pop[i][1][:])
        
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
                child = parents[0][:]
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child, v, k, all_blocks, random_block)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [fitness(ind) for ind in new_population]
        population = new_population
        
    return best_individual


def crossover(parent1, parent2):
    """
    Uniform crossover: pick each block from parent1 or parent2 randomly.
    Ensure distinctness by using a set.
    """
    # Ensure parent1 is the "better" one (shorter or equal length)
    if len(parent1) > len(parent2):
        parent1, parent2 = parent2, parent1
    
    child = []
    seen = set()
    
    # Target size: similar to parent1
    target_size = len(parent1)
    
    # Add blocks from parent1 and parent2 randomly
    # Shuffle indices
    indices1 = list(range(len(parent1)))
    indices2 = list(range(len(parent2)))
    random.shuffle(indices1)
    random.shuffle(indices2)
    
    i1, i2 = 0, 0
    while len(child) < target_size:
        # Prefer parent1
        if i1 < len(parent1) and (i2 >= len(parent2) or random.random() < 0.6):
            b = tuple(parent1[indices1[i1]])
            i1 += 1
        else:
            b = tuple(parent2[indices2[i2]])
            i2 += 1
            
        if b not in seen:
            child.append(list(b))
            seen.add(b)
            
    return child

def mutate(individual, v, k, all_blocks, random_block_func):
    """
    Mutation:
    1. Replace a random block with a new random block.
    2. Add a random block.
    3. Remove a random block.
    """
    n = len(individual)
    if n == 0:
        return individual

    op = random.random()
    
    if op < 0.5:
        # Replace
        idx = random.randint(0, n - 1)
        new_block = random_block_func()
        # Ensure it's not a duplicate of an existing one (simple check)
        # If duplicate, just keep old one or try again?
        # For now, just replace. Duplicates are handled by fitness (size penalty)
        # or we can deduplicate here.
        individual[idx] = new_block
    elif op < 0.75:
        # Add
        new_block = random_block_func()
        individual.append(new_block)
    else:
        # Remove
        if len(individual) > 1:
            idx = random.randint(0, n - 1)
            individual.pop(idx)
            
    # Deduplicate
    seen = set()
    unique_ind = []
    for b in individual:
        tb = tuple(b)
        if tb not in seen:
            unique_ind.append(b)
            seen.add(tb)
            
    return unique_ind
# EVOLVE-BLOCK-END
