import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute t-subsets and map to index
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute t_to_blocks: list of sets of block indices
    # t_to_blocks[i] contains all block indices that cover t_subsets[i]
    t_to_blocks = [set() for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].add(i)

    def greedy_cover(rng):
        """
        Perform a greedy set cover with random tie-breaking.
        Returns a list of block indices.
        """
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains: number of uncovered t-subsets covered by each block
            # We only care about blocks that cover at least one uncovered subset.
            # gains[i] = sum(uncovered[block_t_indices[i]])
            # This is efficient with numpy boolean indexing if we had a flat array,
            # but block_t_indices is 2D. 
            # sum(uncovered[block_t_indices]) works because boolean indexing on the 
            # inner dimension is not directly possible in that shape without flattening.
            # Let's flatten or use a loop?
            # Actually, `uncovered[block_t_indices]` creates a (n_k, n_t_in_k) boolean array.
            # Summing along axis 1 gives the count.
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        return selected

    def local_search(initial_selection):
        """
        Refine a selection using redundancy removal and swaps.
        Returns a refined list of block indices.
        """
        if not initial_selection:
            return []
            
        remaining = list(initial_selection)
        remaining_set = set(remaining)
        
        # Build coverage count
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in remaining:
            coverage_count[block_t_indices[bi]] += 1
            
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # 1. Remove redundant blocks
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    # If all t-subsets covered by bi are covered by >1 block, it's redundant
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        changed = True
                        break # Restart scan as indices changed
                
            if time.time() > deadline - 2.0:
                break

            # 2. Try swaps
            # We iterate through a copy of remaining to avoid index issues during modification
            # But we break after a swap to restart the loop, so we can just iterate.
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                
                bi = remaining[i]
                ts = block_t_indices[bi]
                
                # Find unique t-subsets covered by bi
                unique_ts = ts[coverage_count[ts] == 1]
                
                if len(unique_ts) == 0:
                    # This block is redundant
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    improved = True
                    break # Restart outer loop
                
                # Find candidates B that cover all unique_ts
                # Intersection of t_to_blocks for each t in unique_ts
                # Start with the first one's set
                candidates = set(t_to_blocks[unique_ts[0]])
                for t_idx in unique_ts[1:]:
                    candidates.intersection_update(t_to_blocks[t_idx])
                
                # Filter out blocks already in solution
                candidates.difference_update(remaining_set)
                
                if candidates:
                    # Pick a random candidate
                    new_bi = rng.choice(list(candidates))
                    
                    # Perform swap
                    # Remove bi
                    coverage_count[block_t_indices[bi]] -= 1
                    remaining_set.remove(bi)
                    remaining.pop(i)
                    
                    # Add new_bi
                    coverage_count[block_t_indices[new_bi]] += 1
                    remaining_set.add(new_bi)
                    remaining.append(new_bi)
                    
                    improved = True
                    break # Restart outer loop after a swap to re-evaluate redundancies
            
        return remaining

    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < deadline - 10.0:
        initial = greedy_cover(rng)
        
        if time.time() > deadline - 10.0:
            break
            
        refined = local_search(initial)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined
            
    if best_solution is None:
        # Fallback: if we ran out of time before completing a full cycle,
        # try one quick greedy if possible, or return empty (invalid) but we should avoid invalid.
        # The loop condition should ensure at least one attempt.
        # If deadline was very tight, we might have zero.
        # Let's do a final check.
        if time.time() < deadline - 2.0:
             initial = greedy_cover(rng)
             if initial:
                 best_solution = initial
                 best_size = len(initial)
    
    if best_solution is None:
        return []
        
    return [list(k_subsets[i]) for i in best_solution]
