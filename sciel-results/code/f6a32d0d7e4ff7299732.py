"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))

def _block_to_tset_tuples(block, t):
    """Return the set of t-subsets covered by a single block."""
    return set(itertools.combinations(sorted(block), t))

def _coverage_mask(block, t_index_map, t):
    """Return a set of indices of t-subsets covered by block.
    t_index_map maps a t-subset tuple to its index."""
    covered = set()
    for combo in itertools.combinations(sorted(block), t):
        idx = t_index_map.get(combo)
        if idx is not None:
            covered.add(idx)
    return covered

def _greedy_cover(v, k, t, num_blocks=None):
    """Greedy covering heuristic. Returns list of blocks."""
    all_subsets = _all_t_subsets(v, t)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Use a boolean array for uncovered tracking
    uncovered = np.ones(num_subsets, dtype=bool)
    blocks = []
    
    max_blocks = num_blocks if num_blocks else num_subsets
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
        
        # Pick the uncovered t-subset with the smallest first element (for determinism)
        # Actually, let's just pick the first uncovered one
        first_idx = np.where(uncovered)[0][0]
        first_subset = all_subsets[first_idx]
        
        # Build a block starting with this subset, greedily adding points
        block = list(first_subset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            # Pre-compute: for each candidate point, how many new t-subsets does it cover?
            # A new t-subset is one that contains (t-1) elements from block and the new point
            # and is currently uncovered
            # We need to check all combinations of (t-1) from block
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # For each (t-1)-subset of current block, form a t-subset with x
                for rest in itertools.combinations(block, t - 1):
                    # Sort rest + (x,) to get canonical form
                    new_subset = tuple(sorted(rest + (x,)))
                    idx = t_index_map.get(new_subset)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                # No point adds coverage, just pick an arbitrary unused point
                used = set(block)
                candidates = [x for x in range(v) if x not in used]
                if not candidates:
                    break
                best_point = candidates[0]
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for combo in itertools.combinations(block, t):
            idx = t_index_map.get(combo)
            if idx is not None:
                uncovered[idx] = False
    
    return blocks

def _random_block(v, k):
    """Generate a random k-subset of {0,...,v-1}."""
    return sorted(random.sample(range(v), k))

def _random_covering(v, k, t, num_blocks):
    """Generate a random covering by picking random blocks until covered or limit reached."""
    all_subsets = _all_t_subsets(v, t)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    uncovered = np.ones(num_subsets, dtype=bool)
    blocks = []
    
    for _ in range(num_blocks):
        if not uncovered.any():
            break
        block = _random_block(v, k)
        blocks.append(block)
        for combo in itertools.combinations(block, t):
            idx = t_index_map.get(combo)
            if idx is not None:
                uncovered[idx] = False
    
    return blocks

def _crossover(parent1, parent2, v, k, t, max_blocks):
    """Crossover: merge two coverings by selecting blocks that cover the most uncovered t-subsets."""
    all_subsets = _all_t_subsets(v, t)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Pool of blocks from both parents
    pool = list(parent1) + list(parent2)
    # Remove duplicates
    seen = set()
    unique_pool = []
    for b in pool:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_pool.append(b)
    
    # Greedily select blocks from the pool that cover the most uncovered t-subsets
    uncovered = np.ones(num_subsets, dtype=bool)
    child_blocks = []
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
        
        # Find the block in the remaining pool that covers the most uncovered t-subsets
        best_block = None
        best_count = -1
        best_idx = -1
        
        for i, block in enumerate(unique_pool):
            # Count how many uncovered t-subsets this block covers
            count = 0
            for combo in itertools.combinations(block, t):
                idx = t_index_map.get(combo)
                if idx is not None and uncovered[idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = block
                best_idx = i
        
        if best_block is None or best_count == 0:
            break
        
        child_blocks.append(best_block)
        # Remove from pool
        unique_pool.pop(best_idx)
        
        # Update uncovered
        for combo in itertools.combinations(best_block, t):
            idx = t_index_map.get(combo)
            if idx is not None:
                uncovered[idx] = False
    
    return child_blocks

def _repair(v, k, t, blocks):
    """Repair a partial covering by greedily adding blocks to cover remaining t-subsets."""
    all_subsets = _all_t_subsets(v, t)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Determine which t-subsets are already covered
    covered = np.zeros(num_subsets, dtype=bool)
    for block in blocks:
        for combo in itertools.combinations(sorted(block), t):
            idx = t_index_map.get(combo)
            if idx is not None:
                covered[idx] = True
    
    uncovered = ~covered
    if not uncovered.any():
        return blocks
    
    blocks = list(blocks)
    
    while uncovered.any():
        # Greedy: find the block that covers the most uncovered t-subsets
        # Try all possible k-subsets? That's C(19,7) = 50388, which is feasible
        # But we can be smarter: start from an uncovered t-subset and build up
        
        # Pick first uncovered
        first_idx = np.where(uncovered)[0][0]
        first_subset = all_subsets[first_idx]
        
        # Build a block containing this subset
        block = list(first_subset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_subset = tuple(sorted(rest + (x,)))
                    idx = t_index_map.get(new_subset)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                used = set(block)
                candidates = [x for x in range(v) if x not in used]
                if not candidates:
                    break
                best_point = candidates[0]
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        for combo in itertools.combinations(block, t):
            idx = t_index_map.get(combo)
            if idx is not None:
                uncovered[idx] = False
    
    return blocks

def _is_valid_covering(v, k, t, blocks):
    """Check if the blocks form a valid covering."""
    all_subsets = _all_t_subsets(v, t)
    covered = set()
    for block in blocks:
        for combo in itertools.combinations(sorted(block), t):
            covered.add(combo)
    return len(covered) == len(all_subsets)

def construct(v, k, t):
    """Main entry point: use genetic algorithm to find a covering design."""
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initialize population
    pop_size = 8
    max_blocks_per_individual = 200
    
    # Create initial population: mix of greedy and random
    population = []
    
    # Add a few greedy solutions
    for i in range(3):
        blocks = _greedy_cover(v, k, t, num_blocks=max_blocks_per_individual)
        population.append(blocks)
    
    # Add some random + repaired solutions
    for i in range(pop_size - 3):
        blocks = _random_covering(v, k, t, num_blocks=80)
        blocks = _repair(v, k, t, blocks)
        population.append(blocks)
    
    # Evaluate initial population
    def fitness(blocks):
        return len(blocks)
    
    best_individual = min(population, key=fitness)
    best_fitness = fitness(best_individual)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort population by fitness
        population.sort(key=fitness)
        
        # Elitism: keep the best half
        elite_size = max(1, pop_size // 2)
        elite = population[:elite_size]
        
        # Generate new population
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            # Select parents
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            # Crossover
            child = _crossover(p1, p2, v, k, t, max_blocks_per_individual)
            
            # Repair
            child = _repair(v, k, t, child)
            
            # Mutation: occasionally replace a random block with a random one
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = _random_block(v, k)
                # Re-repair after mutation
                child = _repair(v, k, t, child)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        current_best = min(population, key=fitness)
        current_best_fitness = fitness(current_best)
        if current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness
        
        # Check if we've hit a good enough solution
        if best_fitness <= 151:
            break
    
    # Final repair to ensure validity
    final_blocks = _repair(v, k, t, best_individual)
    
    # Verify validity
    if not _is_valid_covering(v, k, t, final_blocks):
        # Fallback to pure greedy
        final_blocks = _greedy_cover(v, k, t, num_blocks=500)
    
    return final_blocks
# EVOLVE-BLOCK-END
