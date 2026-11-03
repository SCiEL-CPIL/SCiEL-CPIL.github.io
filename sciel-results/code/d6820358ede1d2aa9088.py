import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    # 115s deadline to be safe within 120s limit
    deadline = start_time + 115.0
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map t-subsets within a k-subset to global t-subset indices
    # block_t_indices[i] contains the indices of t-subsets covered by block i
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute for each t-subset, which blocks cover it
    # This is useful for local search swaps
    t_to_blocks = [set() for _ in range(n_t)]
    for i in range(n_k):
        for t_idx in block_t_indices[i]:
            t_to_blocks[t_idx].add(i)
    
    # 2. Greedy Set Cover with Adaptive Weights
    # We maintain a weight for each t-subset. 
    # Gain of a block = sum of weights of uncovered t-subsets it covers.
    # This is a heuristic for Lagrangean relaxation.
    
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # We run multiple restarts. Each restart uses a different random seed for tie-breaking and perturbation.
    # The adaptive weights are reset for each restart, but we can also try different weight update strategies.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        if time.time() > deadline - 10.0:
            break
            
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        weights = np.ones(n_t, dtype=np.float64)
        selected_blocks = []
        selected_set = set()
        
        # Greedy Loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            if not np.any(uncovered):
                break
                
            # Calculate gains: sum of weights for uncovered t-subsets covered by each block
            # gains[i] = sum(weights[block_t_indices[i][uncovered_indices]])
            # Efficient way: 
            # We can't easily vectorize the "sum over subset" part without loops or sparse matrices.
            # However, n_k is ~75582, n_t_in_k is 56. 
            # Let's use a matrix multiplication approach or simple loops if n_k is small enough.
            # Actually, a simple loop over blocks might be too slow in Python.
            # Let's use numpy:
            # Create a mask for uncovered t-subsets in each block row? No, that's hard.
            
            # Alternative: 
            # gains = np.zeros(n_k)
            # for i in range(n_k):
            #    idxs = block_t_indices[i]
            #    gains[i] = weights[idxs][uncovered[idxs]].sum()
            # This is O(n_k * n_t_in_k) which is ~4M operations. In Python loop, this is ~0.5s per iteration.
            # If we need ~300 blocks, that's 150s. Too slow.
            
            # Optimization: 
            # We can maintain an array `current_gains` and update it incrementally?
            # No, weights change globally.
            
            # Let's stick to a faster approximation or a different greedy strategy.
            # Standard greedy (max uncovered count) is O(n_k * n_t_in_k) too.
            # The previous program used `uncovered[block_t_indices].sum(axis=1)`.
            # This creates a temporary array of size (n_k, n_t_in_k) of booleans, then sums.
            # This is vectorized in C and very fast.
            
            # Let's use the standard greedy first, but with a twist:
            # We prioritize blocks that cover rare t-subsets.
            # We can precompute the "rarity" of each t-subset (1/num_blocks_covering_it).
            # And use gain = sum(rarity[t] for t in block if uncovered[t]).
            
            # Let's try a hybrid: 
            # 1. Calculate standard gain (count of uncovered)
            # 2. If there are ties, break by sum of rarities.
            
            # Actually, let's just use the standard vectorized greedy which is fast.
            # gains = uncovered[block_t_indices].sum(axis=1)
            
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
            # Update weights: increase weight of covered t-subsets to make them "expensive" 
            # so future blocks avoid covering them again? 
            # No, in Lagrangean relaxation for set cover, we usually increase penalty for uncovered constraints.
            # Here, we want to cover all. 
            # A common heuristic is to increase the weight of t-subsets that are *still* uncovered 
            # if they are hard to cover? 
            # Let's try: increase weight of t-subsets covered by the selected block.
            # This makes them "satisfied" and reduces the incentive to cover them again (redundancy).
            weights[block_t_indices[best_idx]] += 1.0
            
        # End Greedy Loop
        
        if not selected_blocks:
            continue
            
        # Local Search Phase
        # 1. Remove redundant blocks
        # 2. Swap blocks to reduce size or improve structure
        
        remaining = list(selected_blocks)
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
                    # If all t-subsets in this block are covered by >1 block, it's redundant
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        changed = True
                        break # Restart scan
                
            if time.time() > deadline - 2.0:
                break

            # 2. Try swaps
            # For each block, if it has unique t-subsets, see if another block can cover them
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                    
                bi = remaining[i]
                ts = block_t_indices[bi]
                
                # Find unique t-subsets covered by bi
                unique_ts = ts[coverage_count[ts] == 1]
                
                if len(unique_ts) == 0:
                    # Redundant, should have been caught above, but maybe coverage changed?
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    improved = True
                    break # Restart outer loop
                
                # Find candidates B that cover all unique_ts
                # Intersection of t_to_blocks for each unique t
                # Optimization: Start with the set of blocks covering the "rarest" unique t-subset
                # to minimize intersection size.
                
                # Sort unique_ts by size of t_to_blocks[t]
                # This might be slow, so let's just pick the first one or the one with smallest set size
                min_size = float('inf')
                min_t = unique_ts[0]
                for t_idx in unique_ts:
                    if len(t_to_blocks[t_idx]) < min_size:
                        min_size = len(t_to_blocks[t_idx])
                        min_t = t_idx
                        
                candidates = set(t_to_blocks[min_t])
                for t_idx in unique_ts:
                    if t_idx == min_t:
                        continue
                    candidates.intersection_update(t_to_blocks[t_idx])
                    if not candidates:
                        break
                
                # Filter out blocks already in solution
                candidates.difference_update(remaining_set)
                
                if candidates:
                    # Pick a random candidate
                    new_bi = rng.choice(list(candidates))
                    
                    # Perform swap
                    coverage_count[block_t_indices[bi]] -= 1
                    remaining_set.remove(bi)
                    remaining.pop(i)
                    
                    coverage_count[block_t_indices[new_bi]] += 1
                    remaining_set.add(new_bi)
                    remaining.append(new_bi)
                    
                    improved = True
                    break # Restart outer loop after a swap
        
        # 3. Iterated Local Search (Perturbation)
        # If we have time, perturb the current best solution and repair
        if time.time() < deadline - 15.0 and best_solution is not None:
            # Start from a copy of the current best solution indices
            # We need to store indices for best_solution to avoid conversion overhead
            if not hasattr(construct, '_best_indices'):
                construct._best_indices = None
                
            if construct._best_indices is not None:
                perturbed = list(construct._best_indices)
                perturbed_set = set(perturbed)
                
                # Calculate coverage
                p_coverage = np.zeros(n_t, dtype=np.int32)
                for bi in perturbed:
                    p_coverage[block_t_indices[bi]] += 1
                
                # Delete 5-15% of blocks
                num_del = max(1, int(len(perturbed) * rng.uniform(0.05, 0.15)))
                for _ in range(num_del):
                    if not perturbed:
                        break
                    idx = rng.randrange(len(perturbed))
                    bi = perturbed.pop(idx)
                    perturbed_set.remove(bi)
                    p_coverage[block_t_indices[bi]] -= 1
                
                # Repair using greedy
                while True:
                    if time.time() > deadline - 5.0:
                        break
                    
                    uncovered_mask = p_coverage == 0
                    if not np.any(uncovered_mask):
                        break
                    
                    gains = uncovered_mask[block_t_indices].sum(axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                        
                    candidates = np.where(gains == max_gain)[0]
                    # Filter out already selected
                    valid_candidates = [c for c in candidates if c not in perturbed_set]
                    if not valid_candidates:
                        break
                    
                    best_idx = rng.choice(valid_candidates)
                    perturbed.append(best_idx)
                    perturbed_set.add(best_idx)
                    p_coverage[block_t_indices[best_idx]] += 1
                
                # If repair succeeded, run local search on the new solution
                if not np.any(p_coverage == 0):
                    # Run local search (simplified: just redundancy removal and swaps)
                    improved_p = True
                    while improved_p:
                        if time.time() > deadline - 2.0:
                            break
                        improved_p = False
                        
                        # Remove redundant
                        changed = True
                        while changed:
                            if time.time() > deadline - 2.0:
                                break
                            changed = False
                            for i in range(len(perturbed)):
                                bi = perturbed[i]
                                ts = block_t_indices[bi]
                                if np.all(p_coverage[ts] > 1):
                                    p_coverage[ts] -= 1
                                    perturbed.pop(i)
                                    perturbed_set.remove(bi)
                                    changed = True
                                    break
                        
                        if time.time() > deadline - 2.0:
                            break

                        # Swaps
                        for i in range(len(perturbed)):
                            if time.time() > deadline - 2.0:
                                break
                            
                            bi = perturbed[i]
                            ts = block_t_indices[bi]
                            unique_ts = ts[p_coverage[ts] == 1]
                            
                            if len(unique_ts) == 0:
                                p_coverage[ts] -= 1
                                perturbed.pop(i)
                                perturbed_set.remove(bi)
                                improved_p = True
                                break
                            
                            min_size = float('inf')
                            min_t = unique_ts[0]
                            for t_idx in unique_ts:
                                if len(t_to_blocks[t_idx]) < min_size:
                                    min_size = len(t_to_blocks[t_idx])
                                    min_t = t_idx
                                    
                            candidates = set(t_to_blocks[min_t])
                            for t_idx in unique_ts:
                                if t_idx == min_t:
                                    continue
                                candidates.intersection_update(t_to_blocks[t_idx])
                                if not candidates:
                                    break
                            
                            candidates.difference_update(perturbed_set)
                            
                            if candidates:
                                new_bi = rng.choice(list(candidates))
                                
                                p_coverage[block_t_indices[bi]] -= 1
                                perturbed_set.remove(bi)
                                perturbed.pop(i)
                                
                                p_coverage[block_t_indices[new_bi]] += 1
                                perturbed_set.add(new_bi)
                                perturbed.append(new_bi)
                                
                                improved_p = True
                                break
                                
                    # Check if improved
                    if len(perturbed) < best_size:
                        best_size = len(perturbed)
                        construct._best_indices = list(perturbed)
                        best_solution = [k_subsets[i] for i in perturbed]
        
        # Update global best
        if len(remaining) < best_size:
            best_size = len(remaining)
            construct._best_indices = list(remaining)
            best_solution = [k_subsets[i] for i in remaining]

    if best_solution is None:
        # Fallback: return empty if no solution found (should not happen)
        return []
        
    return [list(b) for b in best_solution]
