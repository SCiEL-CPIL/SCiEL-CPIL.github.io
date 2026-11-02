import itertools
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets, all k-blocks, and block-to-tset mappings."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-sets it covers (as a set of tset indices)
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
    
    # Precompute tset_to_block_indices: for each tset, which blocks cover it
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_block_indices

def _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, time_limit):
    """Fast greedy set cover using precomputed data."""
    deadline = time.time() + time_limit
    uncovered = np.ones(n_tsets, dtype=bool)
    blocks_chosen = []
    
    while np.any(uncovered) and time.time() < deadline:
        # Find uncovered t-sets
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-set
        rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
        
        # Find best block covering this t-set (and maximizing total uncovered coverage)
        best_block = -1
        best_gain = -1
        for b_idx in tset_to_block_indices[rand_idx]:
            gain = 0
            for ts_idx in block_tsets[b_idx]:
                if uncovered[ts_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks_chosen.append(best_block)
        
        # Update uncovered
        for ts_idx in block_tsets[best_block]:
            uncovered[ts_idx] = False
            
    return blocks_chosen

def _local_search(blocks_idx, n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, time_limit):
    """Local search to improve the covering."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    
    # Precompute coverage for current solution
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    
    best = current[:]
    best_uncovered = n_tsets - np.count_nonzero(covered)
    
    # Convert to set for fast membership testing
    current_set = set(current)
    
    while time.time() < deadline:
        improved = False
        
        # Try removing redundant blocks
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            # Check if any t-set is only covered by this block
            redundant = True
            for ts_idx in block_tsets[b_idx]:
                if covered[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                current.pop(i)
                current_set.discard(b_idx)
                for ts_idx in block_tsets[b_idx]:
                    covered[ts_idx] -= 1
                uncovered_count = n_tsets - np.count_nonzero(covered)
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping: replace one block with another
        for i in range(len(current)):
            if time.time() > deadline:
                break
            old_b = current[i]
            # Find t-sets that become uncovered if we remove old_b
            newly_uncovered = []
            for ts_idx in block_tsets[old_b]:
                if covered[ts_idx] == 1:
                    newly_uncovered.append(ts_idx)
            
            if not newly_uncovered:
                # Just remove it (redundant)
                current.pop(i)
                current_set.discard(old_b)
                for ts_idx in block_tsets[old_b]:
                    covered[ts_idx] -= 1
                uncovered_count = n_tsets - np.count_nonzero(covered)
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
            
            # Try to find a replacement block that covers all newly_uncovered t-sets
            # and maximizes additional coverage
            best_new_b = -1
            best_gain = -1
            for b_idx in tset_to_block_indices[newly_uncovered[0]]:
                if b_idx == old_b or b_idx in current_set:
                    continue
                # Check if b_idx covers all newly_uncovered
                covers_all = True
                for ts_idx in newly_uncovered:
                    if ts_idx not in block_tsets[b_idx]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                # Count additional coverage
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if covered[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new_b = b_idx
            
            if best_new_b != -1:
                # Perform swap
                current[i] = best_new_b
                current_set.discard(old_b)
                current_set.add(best_new_b)
                for ts_idx in block_tsets[old_b]:
                    covered[ts_idx] -= 1
                for ts_idx in block_tsets[best_new_b]:
                    covered[ts_idx] += 1
                uncovered_count = n_tsets - np.count_nonzero(covered)
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
        
        if not improved:
            # Try random perturbation to escape local optima
            if time.time() > deadline:
                break
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Randomly swap two blocks
                if len(current) < 2:
                    break
                i, j = rng.sample(range(len(current)), 2)
                old_i, old_j = current[i], current[j]
                
                # Remove both
                for ts_idx in block_tsets[old_i]:
                    covered[ts_idx] -= 1
                for ts_idx in block_tsets[old_j]:
                    covered[ts_idx] -= 1
                current_set.discard(old_i)
                current_set.discard(old_j)
                
                # Pick two random blocks
                new_i, new_j = rng.sample(range(n_blocks), 2)
                current[i] = new_i
                current[j] = new_j
                current_set.add(new_i)
                current_set.add(new_j)
                for ts_idx in block_tsets[new_i]:
                    covered[ts_idx] += 1
                for ts_idx in block_tsets[new_j]:
                    covered[ts_idx] += 1
                
                uncovered_count = n_tsets - np.count_nonzero(covered)
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
    
    return best

def _verify_covering(blocks_idx, block_tsets, n_tsets):
    """Verify that the blocks cover all t-sets."""
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in blocks_idx:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    return np.all(covered > 0)

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit
    
    # Precompute
    all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_block_indices = _precompute(v, k, t)
    
    # Multiple greedy runs with different seeds
    best_blocks = None
    best_size = float('inf')
    
    for seed in range(10):
        if time.time() - start_time > total_budget - 30.0:
            break
        rng = random.Random(seed)
        blocks_idx = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, 5.0)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx[:]
    
    if best_blocks is None:
        # Fallback: just return first greedy result
        rng = random.Random(42)
        best_blocks = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, 10.0)
        best_size = len(best_blocks)
    
    # Local search on best solution
    if time.time() - start_time < total_budget - 5.0:
        rng = random.Random(123)
        remaining_time = min(60.0, total_budget - (time.time() - start_time) - 5.0)
        improved = _local_search(best_blocks, n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Try to reduce further with more aggressive local search
    if time.time() - start_time < total_budget - 5.0 and best_size > 70:
        rng = random.Random(456)
        remaining_time = min(30.0, total_budget - (time.time() - start_time) - 5.0)
        improved = _local_search(best_blocks, n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final verification
    if not _verify_covering(best_blocks, block_tsets, n_tsets):
        # Something went wrong; rebuild
        rng = random.Random(789)
        best_blocks = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, rng, 10.0)
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result

# EVOLVE-BLOCK-END
