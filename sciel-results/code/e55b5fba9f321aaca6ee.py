"""Genetic algorithm covering design C(19,7,4) with custom crossover."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict

# Global cache for precomputed data
_CACHE = {}

def _get_precomputed(v, k, t):
    """Get or compute precomputed data for the given parameters."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # For each k-block, precompute which t-subsets it covers
    # Use a list of lists: block_covered_subsets[i] = list of t-subset indices covered by block i
    block_covered = []
    for block in all_k_blocks:
        covered = []
        for combo in itertools.combinations(block, t):
            covered.append(t_to_idx[combo])
        block_covered.append(covered)
    
    # Also build reverse mapping: for each t-subset, which k-blocks cover it
    # This helps with efficient repair
    subset_to_blocks = defaultdict(list)
    for i, covered in enumerate(block_covered):
        for j in covered:
            subset_to_blocks[j].append(i)
    
    # Convert to numpy arrays for fast operations
    # coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_k, num_t), dtype=np.uint8)
    for i, covered in enumerate(block_covered):
        for j in covered:
            coverage_matrix[i, j] = 1
    
    data = {
        'all_t_subsets': all_t_subsets,
        't_to_idx': t_to_idx,
        'num_t': num_t,
        'all_k_blocks': all_k_blocks,
        'num_k': num_k,
        'block_covered': block_covered,
        'subset_to_blocks': subset_to_blocks,
        'coverage_matrix': coverage_matrix,
    }
    
    _CACHE[key] = data
    return data

def _greedy_cover(v, k, t, time_limit=15.0, random_tie_break=False):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_t = data['num_t']
    num_k = data['num_k']
    
    uncovered = np.ones(num_t, dtype=np.uint8)
    selected_blocks = []
    used_blocks = np.zeros(num_k, dtype=bool)
    
    while uncovered.any() and time.time() - start_time < time_limit:
        gains = coverage_matrix @ uncovered
        
        # Mask out already used blocks
        gains[used_blocks] = 0
        
        best_gain = int(np.max(gains))
        if best_gain == 0:
            break
        
        if random_tie_break:
            # Among blocks with max gain, pick randomly
            max_indices = np.where(gains == best_gain)[0]
            best_idx = int(random.choice(max_indices))
        else:
            best_idx = int(np.argmax(gains))
        
        selected_blocks.append(best_idx)
        used_blocks[best_idx] = True
        uncovered[coverage_matrix[best_idx] == 1] = 0
    
    return selected_blocks

def _repair_cover(v, k, t, block_indices, time_limit=10.0):
    """Repair a list of block indices to ensure full coverage."""
    if not block_indices:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_t = data['num_t']
    
    # Build current coverage mask
    covered = np.zeros(num_t, dtype=bool)
    for bi in block_indices:
        covered[coverage_matrix[bi] == 1] = True
    
    uncovered = ~covered
    if not uncovered.any():
        return list(block_indices)
    
    # Add blocks to cover remaining gaps
    uncovered_int = uncovered.astype(np.uint8)
    result_blocks = list(block_indices)
    used_set = set(block_indices)
    
    while uncovered.any() and time.time() - start_time < time_limit:
        gains = coverage_matrix @ uncovered_int
        # Mask out already used
        for bi in used_set:
            gains[bi] = 0
        
        best_gain = int(np.max(gains))
        if best_gain == 0:
            break
        
        best_idx = int(np.argmax(gains))
        result_blocks.append(best_idx)
        used_set.add(best_idx)
        uncovered[coverage_matrix[best_idx] == 1] = False
        uncovered_int = uncovered.astype(np.uint8)
    
    # Reduction phase: iteratively remove redundant blocks
    if len(result_blocks) <= 1:
        return result_blocks
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_blocks = []
        for i, bi in enumerate(result_blocks):
            if len(result_blocks) <= 1:
                new_blocks.append(bi)
                continue
            
            # Check if removing this block still covers everything
            temp_covered = np.zeros(num_t, dtype=bool)
            for j, bj in enumerate(result_blocks):
                if j != i:
                    temp_covered[coverage_matrix[bj] == 1] = True
            
            if temp_covered.all():
                changed = True
            else:
                new_blocks.append(bi)
        
        if changed:
            result_blocks = new_blocks
    
    return result_blocks

def _crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """Custom crossover: merge two coverings by greedy selection from union,
    prioritizing blocks that cover the most uncovered 4-subsets."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_t = data['num_t']
    
    # Build union of blocks from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        if b not in seen:
            seen.add(b)
            pool_indices.append(b)
    
    # Greedy selection from pool, prioritizing coverage of uncovered subsets
    uncovered = np.ones(num_t, dtype=np.uint8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        best_gain = -1
        best_pos = -1
        
        for pos, bi in enumerate(available):
            gain = int(np.sum(coverage_matrix[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        
        if best_pos == -1 or best_gain == 0:
            break
        
        child_blocks.append(available[best_pos])
        uncovered[coverage_matrix[available[best_pos]] == 1] = 0
        available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair_cover(v, k, t, child_blocks, time_limit=time_limit)
    
    return child_blocks

def _local_search(v, k, t, block_indices, time_limit=8.0):
    """Local search: try removing, swapping, and adding blocks to find smaller coverings."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_t = data['num_t']
    num_k = data['num_k']
    
    current = list(block_indices)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing each block and repairing
        for i in range(len(current)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current[:i] + current[i+1:]
            candidate = _repair_cover(v, k, t, candidate, time_limit=3.0)
            
            if len(candidate) < len(current):
                current = candidate
                improved = True
                break
        
        if not improved:
            # Try swap operations: replace one block with another
            for i in range(len(current)):
                if time.time() - start_time >= time_limit:
                    break
                
                # Find blocks that could replace current[i]
                # Prefer blocks that cover subsets not well-covered by others
                temp_covered = np.zeros(num_t, dtype=bool)
                for j, bj in enumerate(current):
                    if j != i:
                        temp_covered[coverage_matrix[bj] == 1] = True
                
                # Subsets only covered by block i
                unique_to_i = coverage_matrix[current[i]] & ~temp_covered
                if not unique_to_i.any():
                    continue
                
                # Find replacement blocks that cover these unique subsets
                gains = coverage_matrix @ unique_to_i.astype(np.uint8)
                gains[current] = 0  # Don't swap with existing blocks
                gains[current[i]] = 0
                
                best_gain = int(np.max(gains))
                if best_gain > 0:
                    replacement = int(np.argmax(gains))
                    candidate = current[:i] + [replacement] + current[i+1:]
                    candidate = _repair_cover(v, k, t, candidate, time_limit=3.0)
                    
                    if len(candidate) < len(current):
                        current = candidate
                        improved = True
                        break
            
            if not improved:
                # Try random perturbations
                for _ in range(3):
                    if time.time() - start_time >= time_limit or not current:
                        break
                    
                    if random.random() < 0.5 and len(current) > 1:
                        # Remove a random block
                        idx_to_remove = random.randint(0, len(current) - 1)
                        candidate = current[:idx_to_remove] + current[idx_to_remove+1:]
                    else:
                        # Replace a random block with a random one
                        idx_to_replace = random.randint(0, len(current) - 1)
                        new_block = random.randint(0, num_k - 1)
                        candidate = current[:idx_to_replace] + [new_block] + current[idx_to_replace+1:]
                    
                    candidate = _repair_cover(v, k, t, candidate, time_limit=3.0)
                    
                    if len(candidate) < len(current):
                        current = candidate
                        improved = True
                        break
                
                if not improved:
                    break
    
    return current

def construct(v, k, t):
    """Main entry point: genetic algorithm with custom crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    num_k = data['num_k']
    
    # Initial population: diverse starting points
    population = []
    
    # 1. Standard greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=15.0, random_tie_break=False)
    population.append(greedy1)
    
    # 2. Greedy with random tie-breaking
    for _ in range(3):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0, random_tie_break=True)
        population.append(greedy_r)
    
    # 3. Random + repair
    for _ in range(3):
        rand_blocks = [random.randint(0, num_k - 1) for _ in range(80)]
        rand_repaired = _repair_cover(v, k, t, rand_blocks, time_limit=10.0)
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
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = random.randint(0, num_k - 1)
                child = _repair_cover(v, k, t, child, time_limit=8.0)
            
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
    final_indices = _repair_cover(v, k, t, best_individual, time_limit=10.0)
    
    # Convert indices back to blocks
    final_blocks = [list(data['all_k_blocks'][bi]) for bi in final_indices]
    
    # Validate coverage
    all_t_subsets = data['all_t_subsets']
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
    
    if len(covered) != len(all_t_subsets):
        # Fallback to greedy
        final_indices = _greedy_cover(v, k, t, time_limit=110.0)
        final_blocks = [list(data['all_k_blocks'][bi]) for bi in final_indices]
    
    return final_blocks
# EVOLVE-BLOCK-END
