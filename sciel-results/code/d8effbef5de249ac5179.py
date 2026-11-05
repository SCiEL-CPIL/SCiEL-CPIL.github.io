"""Genetic algorithm covering design C(19,7,4) with optimized repair and crossover."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Global cache for precomputed data
_CACHE = {}

def _get_precomputed(v, k, t):
    """Get or compute precomputed data for the given parameters."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    # Use uint8 for efficiency
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.uint8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    # Store block tuples for easy access
    block_tuples = [tuple(b) for b in all_blocks]
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
    }
    
    _CACHE[key] = data
    return data

def _blocks_to_indices(blocks, v, k, t):
    """Convert list of blocks to list of indices in all_blocks."""
    data = _get_precomputed(v, k, t)
    indices = []
    for b in blocks:
        # Assume blocks are sorted lists of ints
        bi = -1
        # Fast lookup: use a dictionary mapping tuple to index? 
        # For 19 choose 7 = 50388 blocks, a dict is better.
        if 'block_to_idx' not in data:
             data['block_to_idx'] = {tuple(b): i for i, b in enumerate(data['all_blocks'])}
        bi = data['block_to_idx'].get(tuple(sorted(b)), -1)
        if bi != -1:
            indices.append(bi)
    return indices

def _greedy_cover(v, k, t, time_limit=15.0, random_seed=None):
    """Fast greedy covering using precomputed coverage matrix.
    If random_seed is provided, tie-breaking is random."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = None

    uncovered = np.ones(num_subsets, dtype=np.uint8)
    blocks = []
    
    while uncovered.any():
        gains = coverage_matrix @ uncovered
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        if rng is not None:
            # Find all indices with max_gain
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(rng.choice(candidates))
        else:
            best_idx = int(np.argmax(gains))
        
        blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        
        if time.time() - start_time > time_limit:
            break
    
    return blocks

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce using count-based method."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Convert to indices
    block_indices = _blocks_to_indices(blocks, v, k, t)
    
    # Initialize counts and total_mask
    counts = np.zeros(num_subsets, dtype=np.int32)
    total_mask = np.zeros(num_subsets, dtype=bool)
    
    for bi in block_indices:
        vec = coverage_matrix[bi]
        total_mask[vec == 1] = True
        counts[vec == 1] += 1
    
    # Phase 1: Add missing blocks
    uncovered = ~total_mask
    if uncovered.any():
        uncovered_uint8 = uncovered.astype(np.uint8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_uint8
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            block_indices.append(best_idx)
            vec = coverage_matrix[best_idx]
            total_mask[vec == 1] = True
            counts[vec == 1] += 1
            uncovered = ~total_mask
            uncovered_uint8 = uncovered.astype(np.uint8)
    
    # Phase 2: Reduction
    if len(block_indices) <= 1:
        return [list(data['block_tuples'][bi]) for bi in block_indices]
    
    # Sort indices to try removing less "critical" blocks first?
    # Heuristic: blocks covering fewer unique subsets (where count==1) are easier to remove.
    # But calculating "uniqueness" is expensive. Let's just iterate.
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_indices = []
        for i, bi in enumerate(block_indices):
            if time.time() - start_time >= time_limit:
                new_indices.append(bi)
                continue
                
            vec = coverage_matrix[bi]
            # Check if any subset covered by this block is covered ONLY by this block
            # i.e., counts[vec==1] == 1
            # If so, we cannot remove it.
            # If all counts[vec==1] > 1, we can remove it.
            
            # Optimization: Only check the subsets this block covers
            # mask = vec == 1
            # if np.all(counts[mask] > 1):
            #     # Remove
            #     counts[mask] -= 1
            #     changed = True
            # else:
            #     new_indices.append(bi)
            
            # Vectorized check
            mask = vec == 1
            if np.all(counts[mask] > 1):
                counts[mask] -= 1
                changed = True
            else:
                new_indices.append(bi)
        
        block_indices = new_indices
    
    # Reconstruct blocks
    # Remove duplicates if any (shouldn't happen if input was unique, but just in case)
    unique_indices = list(dict.fromkeys(block_indices))
    result = [list(data['block_tuples'][bi]) for bi in unique_indices]
    
    return result

def _crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """Custom crossover: merge two coverings by greedy selection from union."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build union of blocks from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        bi_list = _blocks_to_indices([b], v, k, t)
        if bi_list:
            bi = bi_list[0]
            if bi not in seen:
                seen.add(bi)
                pool_indices.append(bi)
    
    if not pool_indices:
        return _greedy_cover(v, k, t, time_limit=time_limit)

    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.uint8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        # We need to extract the submatrix for available blocks
        # This is O(|available| * num_subsets) which is fine for small pools
        if len(available) > 0:
             pool_matrix = coverage_matrix[available]
             gains = pool_matrix @ uncovered
             best_pos = int(np.argmax(gains))
             best_gain = gains[best_pos]
             
             if best_gain == 0:
                 break
                 
             best_idx = available[best_pos]
             child_blocks.append(list(data['block_tuples'][best_idx]))
             uncovered[coverage_matrix[best_idx] == 1] = 0
             available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks, time_limit=time_limit)
    
    return child_blocks

def _local_search(v, k, t, blocks, time_limit=8.0):
    """Local search: try removing blocks and repairing to find smaller coverings."""
    start_time = time.time()
    current_blocks = [list(b) for b in blocks]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing each block and see if repair yields a smaller set
        # To save time, try a random subset of blocks first
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_blocks[:i] + current_blocks[i+1:]
            candidate = _repair(v, k, t, candidate, time_limit=2.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                improved = True
                break
        
        if not improved:
            # Try random perturbations (swap)
            for _ in range(5):
                if time.time() - start_time >= time_limit or not current_blocks:
                    break
                
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                # Replace with a random block
                new_block = sorted(random.sample(range(v), k))
                candidate = current_blocks[:idx_to_remove] + [new_block] + current_blocks[idx_to_remove+1:]
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if not improved:
                break
    
    return current_blocks

def construct(v, k, t):
    """Main entry point: genetic algorithm with custom crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    
    # Initial population: diverse starting points
    population = []
    
    # 1. Standard greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=10.0)
    population.append(greedy1)
    
    # 2. Greedy with random tie-breaking (run multiple times)
    for i in range(5):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0, random_seed=i*1000)
        population.append(greedy_r)
    
    # 3. Random + repair (less effective but adds diversity)
    for i in range(2):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(100)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=10.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    
    # Genetic algorithm loop
    pop_size = 10
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 15:
        # Sort by size
        population.sort(key=len)
        
        # Elitism: keep top half
        elite_count = max(3, len(population) // 2)
        elite = population[:elite_count]
        
        new_population = list(elite)
        
        # Generate children via crossover
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_limit - 10:
                break
            
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t, time_limit=8.0)
            
            # Mutation: occasionally replace a block with a random one
            if random.random() < 0.4 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=8.0)
            
            # Local search
            if time.time() - start_time < time_limit - 10:
                child = _local_search(v, k, t, child, time_limit=5.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
    
    # Final repair and validation
    final_blocks = _repair(v, k, t, best_individual, time_limit=10.0)
    
    # Validate coverage
    all_subsets = data['all_subsets']
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
    
    if len(covered) != len(all_subsets):
        # Fallback to greedy
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
    
    return final_blocks
# EVOLVE-BLOCK-END
