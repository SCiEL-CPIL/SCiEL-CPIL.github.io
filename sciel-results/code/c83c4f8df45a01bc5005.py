"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Global cache for precomputed data
_CACHE = {}

def _precompute(v, k, t):
    """Precompute all blocks and their coverage masks."""
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
        
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage masks for all blocks
    # Shape: (num_blocks, num_subsets)
    # Using bool array. 50388 * 3876 * 1 byte ~ 195 MB.
    # If memory is tight, we could use bitsets, but let's try bool first.
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=bool)
    
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = subset_to_idx[combo]
            coverage_matrix[i, idx] = True
            
    _CACHE[(v, k, t)] = (all_blocks, coverage_matrix, all_subsets, subset_to_idx)
    return _CACHE[(v, k, t)]

def _greedy_cover(v, k, t, time_limit=5.0, initial_blocks=None):
    """Fast greedy covering using precomputed masks."""
    start_time = time.time()
    all_blocks, coverage_matrix, all_subsets, subset_to_idx = _precompute(v, k, t)
    num_subsets = len(all_subsets)
    
    if initial_blocks:
        # Start with given blocks
        blocks = list(initial_blocks)
        # Compute current coverage
        covered_mask = np.zeros(num_subsets, dtype=bool)
        # Map block tuples to indices
        block_indices = []
        for b in blocks:
            idx = all_blocks.index(tuple(b)) # This is slow. Better to precompute a dict?
            # For speed in this function, we assume initial_blocks are from the pool or we just recompute mask.
            # Let's just recompute mask for initial blocks to be safe and fast enough for small N.
            for combo in itertools.combinations(sorted(b), t):
                idx_sub = subset_to_idx[combo]
                covered_mask[idx_sub] = True
    else:
        blocks = []
        covered_mask = np.zeros(num_subsets, dtype=bool)
        
    uncovered_count = num_subsets
    uncovered_mask = ~covered_mask
    
    # Set of indices of blocks already used to avoid duplicates
    used_indices = set()
    
    # Precompute a mapping from block tuple to index for fast lookup if we need to check usage
    # But we can just check the `used_indices` set.
    
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
            
        # Find block with max gain
        # gain[i] = sum(coverage_matrix[i] & uncovered_mask)
        # We can compute this with matrix multiplication if we cast to float, but bool & bool sum is faster in numpy?
        # Actually, `coverage_matrix @ uncovered_mask.astype(np.float32)` is very fast.
        gains = coverage_matrix @ uncovered_mask.astype(np.float32)
        
        # Mask out already used blocks
        for idx in used_indices:
            gains[idx] = 0
            
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if uncovered_count > 0 and we have all blocks
            break
            
        blocks.append(list(all_blocks[best_idx]))
        used_indices.add(best_idx)
        
        # Update uncovered mask
        newly_covered = coverage_matrix[best_idx] & uncovered_mask
        uncovered_count -= np.sum(newly_covered)
        uncovered_mask[coverage_matrix[best_idx]] = False
        
    return blocks

def _local_search(v, k, t, blocks, time_limit=5.0):
    """Improve a covering by removing redundant blocks and swapping blocks."""
    start_time = time.time()
    all_blocks, coverage_matrix, all_subsets, subset_to_idx = _precompute(v, k, t)
    num_subsets = len(all_subsets)
    
    if not blocks:
        return []
        
    # Convert blocks to indices for fast mask access
    # We need to map block tuple to index in all_blocks
    # Building a dict for all 50k blocks is fast.
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    current_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b))
        if idx is not None:
            current_indices.append(idx)
        else:
            # If block is not in precomputed list (shouldn't happen if valid), 
            # we might have issues. But construct ensures valid blocks.
            pass
            
    if not current_indices:
        return []
        
    # Compute total coverage
    total_mask = np.zeros(num_subsets, dtype=bool)
    for idx in current_indices:
        total_mask |= coverage_matrix[idx]
        
    num_covered = np.sum(total_mask)
    if num_covered < num_subsets:
        # Invalid cover, repair first? 
        # The GA should only pass valid covers. 
        # If not, we can add greedy blocks.
        blocks = _greedy_cover(v, k, t, time_limit=1.0, initial_blocks=blocks)
        return _local_search(v, k, t, blocks, time_limit=0.0) # Recurse with 0 time to just cleanup? No, just return.
        # Actually, let's just return the greedy result if invalid.
        return blocks

    # 1. Deletion Pass
    # Try removing each block
    changed = True
    while changed and (time.time() - start_time < time_limit):
        changed = False
        new_indices = []
        for i in range(len(current_indices)):
            if time.time() - start_time > time_limit:
                break
            # Check if removing current_indices[i] leaves all covered
            # We need to know if any subset is covered ONLY by this block.
            # This is expensive to check directly.
            # Alternative: 
            # Maintain a count of how many blocks cover each subset?
            # Let's use a different strategy: 
            # Just try removing it and checking coverage? O(N * NumSubsets).
            # N~150, NumSubsets~4000. 150*4000 = 600k ops. Very fast.
            
            # Optimization: 
            # If we remove block i, is there any subset s such that:
            # 1. s is covered by block i
            # 2. s is NOT covered by any other block in current_indices \ {i}
            
            # We can precompute "unique coverage" for each block relative to the current set?
            # That's hard.
            
            # Simple check:
            # Temporarily remove i
            # Compute coverage of others?
            # We can maintain a `coverage_counts` array.
            
            # Let's implement the count-based deletion.
            # This is more efficient if we maintain the counts.
            pass
        
        # Let's restart the local search with a simpler, robust deletion method:
        # Iterate through blocks. For each, check if it's redundant.
        # Redundant if: for all s in block_i, exists j != i such that s in block_j.
        
        # Let's use the count array approach which is O(N * C(k,t)) to update.
        counts = np.zeros(num_subsets, dtype=np.int32)
        for idx in current_indices:
            counts[coverage_matrix[idx]] += 1
            
        new_indices = []
        for idx in current_indices:
            # Check if any subset has count == 1 and is covered by this block
            # If so, this block is essential.
            # If not, it's redundant.
            block_mask = coverage_matrix[idx]
            # Subsets covered by this block:
            # We need to check if counts[s] == 1 for any s where block_mask[s] is True.
            # If yes, we cannot remove it.
            # If no, we can remove it.
            
            # Fast check:
            # essential = np.any(counts[block_mask] == 1)
            if not np.any(counts[block_mask] == 1):
                # Redundant, remove it
                counts[block_mask] -= 1
                changed = True
            else:
                new_indices.append(idx)
                
        current_indices = new_indices
        
    # 2. Replacement Pass (Swapping)
    # Try to replace a block with another that covers the same or more unique subsets?
    # Or simply: Try to find a block that allows removal of another block?
    # This is complex. Let's stick to deletion and a simple "swap with random" if it improves density?
    # Actually, a good heuristic is:
    # For each block, try replacing it with a random block. If the new cover is still valid, keep it?
    # No, we want to minimize size. So we only replace if it allows subsequent deletion?
    
    # Let's try a "2-opt" style move:
    # Pick a block A. Pick a random block B not in set.
    # Replace A with B.
    # Run deletion pass.
    # If size decreased, keep. Else revert.
    
    # This is expensive. Let's just do deletion for now, it's very effective.
    # The crossover will handle exploration.
    
    return [list(all_blocks[i]) for i in current_indices]

def _crossover(parent1, parent2, v, k, t, time_limit=2.0):
    """Merge two coverings by selecting best blocks from union."""
    start_time = time.time()
    all_blocks, coverage_matrix, all_subsets, subset_to_idx = _precompute(v, k, t)
    num_subsets = len(all_subsets)
    
    # Get unique blocks from parents
    pool_tuples = set([tuple(b) for b in parent1 + parent2])
    
    # Map to indices
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    pool_indices = [block_to_idx[t] for t in pool_tuples if t in block_to_idx]
    
    if not pool_indices:
        return []
        
    uncovered_mask = np.ones(num_subsets, dtype=bool)
    child_indices = []
    used = set()
    
    # Greedy selection from pool
    while uncovered_mask.any():
        if time.time() - start_time > time_limit:
            break
            
        best_idx = -1
        best_gain = -1
        
        # Compute gains for all pool blocks
        # We can vectorize this if we create a sub-matrix
        # But pool is small (~300). Loop is fine.
        for idx in pool_indices:
            if idx in used:
                continue
            gain = np.sum(coverage_matrix[idx] & uncovered_mask)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx == -1 or best_gain == 0:
            break
            
        child_indices.append(best_idx)
        used.add(best_idx)
        uncovered_mask[coverage_matrix[best_idx]] = False
        
    # If not fully covered, repair
    if uncovered_mask.any():
        # Add greedy blocks
        # We need to continue greedy from the current state
        # _greedy_cover doesn't support "start with these blocks and continue".
        # Let's just run a quick greedy on the remaining uncovered parts?
        # Or just call _greedy_cover with initial_blocks?
        # My _greedy_cover implementation above handles initial_blocks.
        
        child_blocks = [list(all_blocks[i]) for i in child_indices]
        child_blocks = _greedy_cover(v, k, t, time_limit=1.0, initial_blocks=child_blocks)
        # Convert back to indices for consistency? No, return blocks.
        return child_blocks
        
    return [list(all_blocks[i]) for i in child_indices]

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    _precompute(v, k, t)
    
    # Initial Population
    pop_size = 20
    population = []
    
    # 1. Greedy
    g1 = _greedy_cover(v, k, t, time_limit=5.0)
    g1 = _local_search(v, k, t, g1, time_limit=5.0)
    population.append(g1)
    
    # 2. Random + Repair
    for _ in range(pop_size - 1):
        # Start with some random blocks
        rand_blocks = [list(b) for b in random.sample(list(itertools.combinations(range(v), k)), 50)]
        r1 = _greedy_cover(v, k, t, time_limit=2.0, initial_blocks=rand_blocks)
        r1 = _local_search(v, k, t, r1, time_limit=2.0)
        population.append(r1)
        
    best_individual = min(population, key=len)
    
    while time.time() - start_time < time_limit:
        # Sort population
        population.sort(key=len)
        
        # Elitism
        elite_size = max(2, pop_size // 4)
        new_population = population[:elite_size]
        
        # Generate children
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = random.choice(population[:pop_size//2])
            p2 = random.choice(population[:pop_size//2])
            
            child = _crossover(p1, p2, v, k, t, time_limit=1.0)
            
            # Mutation: Swap a block
            if child and random.random() < 0.3:
                idx = random.randint(0, len(child)-1)
                # Replace with a random block
                new_block = list(random.choice(list(itertools.combinations(range(v), k))))
                child[idx] = new_block
                # Repair
                child = _local_search(v, k, t, child, time_limit=1.0)
                
            new_population.append(child)
            
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
                
    # Final polish
    final_blocks = _local_search(v, k, t, best_individual, time_limit=5.0)
    
    # Validation
    all_subsets, _ = _precompute(v, k, t)[2], None
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
            
    if len(covered) != len(all_subsets):
        # Fallback
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
        
    return final_blocks
# EVOLVE-BLOCK-END
