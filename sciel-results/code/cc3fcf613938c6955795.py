"""Genetic algorithm covering design C(19,7,4) with efficient custom crossover."""
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
    
    # Precompute all k-blocks and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    # Direct mapping from frozenset to block index
    block_to_idx = {}
    for i, b in enumerate(all_blocks):
        block_to_idx[frozenset(b)] = i
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_to_idx': block_to_idx,
    }
    
    _CACHE[key] = data
    return data

def _block_to_idx_fast(block, data):
    """Convert a block to its index using direct mapping."""
    return data['block_to_idx'].get(frozenset(block), -1)

def _greedy_cover(v, k, t, time_limit=15.0):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    uncovered = np.ones(num_subsets, dtype=np.int8)
    blocks = []
    
    while uncovered.any():
        gains = coverage_matrix @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        blocks.append(list(data['all_blocks'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        
        if time.time() - start_time > time_limit:
            break
    
    return blocks

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build current coverage count (how many times each subset is covered)
    coverage_count = np.zeros(num_subsets, dtype=np.int16)
    block_indices = []
    for b in blocks:
        bi = _block_to_idx_fast(b, data)
        if bi >= 0:
            block_indices.append(bi)
            coverage_count += coverage_matrix[bi]
    
    uncovered = coverage_count == 0
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        uncovered_int = uncovered.astype(np.int8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            blocks.append(list(data['all_blocks'][best_idx]))
            block_indices.append(best_idx)
            coverage_count += coverage_matrix[best_idx]
            uncovered = coverage_count == 0
            uncovered_int = uncovered.astype(np.int8)
    
    # Reduction phase: iteratively remove redundant blocks using coverage counts
    if len(blocks) <= 1:
        return blocks
    
    # Recompute block_indices after potential additions
    block_indices = []
    for b in blocks:
        bi = _block_to_idx_fast(b, data)
        if bi >= 0:
            block_indices.append(bi)
    
    # Rebuild coverage count from scratch for accuracy
    coverage_count = np.zeros(num_subsets, dtype=np.int16)
    for bi in block_indices:
        coverage_count += coverage_matrix[bi]
    
    # Try to remove each block if redundant (coverage count stays >= 1)
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_block_indices = []
        for i, bi in enumerate(block_indices):
            if len(block_indices) <= 1:
                new_block_indices.append(bi)
                continue
            
            # Check if removing this block still covers everything
            coverage_count -= coverage_matrix[bi]
            if np.all(coverage_count >= 1):
                # Block is redundant, skip it
                changed = True
            else:
                # Block is needed, keep it
                coverage_count += coverage_matrix[bi]
                new_block_indices.append(bi)
        
        if changed:
            block_indices = new_block_indices
    
    # Reconstruct blocks from block_indices
    unique_indices = list(dict.fromkeys(block_indices))  # preserve order, remove dupes
    result = [list(data['all_blocks'][bi]) for bi in unique_indices]
    
    return result

def _crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """Custom crossover: merge two coverings by greedy selection from union,
    prioritizing blocks unique to each parent."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build union of blocks from both parents, tracking which parent each came from
    seen = set()
    pool_indices = []
    p1_set = set()
    p2_set = set()
    
    for b in parent1:
        bi = _block_to_idx_fast(b, data)
        if bi >= 0:
            p1_set.add(bi)
    
    for b in parent2:
        bi = _block_to_idx_fast(b, data)
        if bi >= 0:
            p2_set.add(bi)
    
    # Blocks unique to p1, unique to p2, and shared
    unique_p1 = p1_set - p2_set
    unique_p2 = p2_set - p1_set
    shared = p1_set & p2_set
    
    # Priority: unique blocks first (they provide diversity), then shared
    pool_indices = list(unique_p1) + list(unique_p2) + list(shared)
    
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.int8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks using vectorized operation
        if available:
            avail_mask = np.zeros(data['num_blocks'], dtype=np.int8)
            for bi in available:
                avail_mask[bi] = 1
            
            # Only consider available blocks
            gains = (coverage_matrix * avail_mask[:, np.newaxis]) @ uncovered
            best_idx = int(np.argmax(gains))
            best_gain = int(gains[best_idx])
            
            if best_gain == 0:
                break
            
            child_blocks.append(list(data['all_blocks'][best_idx]))
            uncovered[coverage_matrix[best_idx] == 1] = 0
            available.remove(best_idx)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks, time_limit=time_limit)
    
    return child_blocks

def _local_search(v, k, t, blocks, time_limit=8.0):
    """Local search: try removing blocks, pairwise swaps, and random perturbations."""
    start_time = time.time()
    current_blocks = [list(b) for b in blocks]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try removing each block and see if repair yields a smaller set
        for i in range(len(current_blocks)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_blocks[:i] + current_blocks[i+1:]
            candidate = _repair(v, k, t, candidate, time_limit=2.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try pairwise block swaps
        for i in range(len(current_blocks)):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 1, len(current_blocks)):
                if time.time() - start_time >= time_limit:
                    break
                
                # Swap blocks i and j with random blocks
                candidate = [list(b) for b in current_blocks]
                candidate[i] = sorted(random.sample(range(v), k))
                candidate[j] = sorted(random.sample(range(v), k))
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 3: Try random perturbations
        for _ in range(5):
            if time.time() - start_time >= time_limit or not current_blocks:
                break
            
            num_perturb = max(1, len(current_blocks) // 15)
            candidate = [list(b) for b in current_blocks]
            for _ in range(num_perturb):
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = sorted(random.sample(range(v), k))
            
            candidate = _repair(v, k, t, candidate, time_limit=3.0)
            
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
    greedy1 = _greedy_cover(v, k, t, time_limit=15.0)
    population.append(greedy1)
    
    # 2. Greedy with random perturbations
    for _ in range(4):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0)
        if greedy_r:
            perturbed = [list(b) for b in greedy_r]
            num_perturb = max(1, len(perturbed) // 12)
            for _ in range(num_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed[idx] = sorted(random.sample(range(v), k))
            perturbed = _repair(v, k, t, perturbed, time_limit=8.0)
            population.append(perturbed)
    
    # 3. Random + repair
    for _ in range(3):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(90)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=8.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    
    # Genetic algorithm loop
    pop_size = 8
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 20:
        # Sort by size
        population.sort(key=len)
        
        # Elitism: keep top half
        elite_count = max(2, len(population) // 2)
        elite = population[:elite_count]
        
        new_population = list(elite)
        
        # Generate children via crossover
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_limit - 15:
                break
            
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t, time_limit=6.0)
            
            # Mutation: occasionally replace a block with a random one
            if random.random() < 0.4 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=5.0)
            
            # Local search
            if time.time() - start_time < time_limit - 15:
                child = _local_search(v, k, t, child, time_limit=4.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
    
    # Final repair and validation
    final_blocks = _repair(v, k, t, best_individual, time_limit=8.0)
    
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
