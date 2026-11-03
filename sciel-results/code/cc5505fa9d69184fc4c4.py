import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    # Optimization: Use a flat array for block_t_indices for faster indexing
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # 4. Calculate Weights for Weighted Greedy
    # Frequency of each t-subset across all k-blocks
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)
    
    # 5. Build sub_to_blocks for local search
    # sub_to_blocks[j] is a list of block indices that cover t-subset j
    sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx in range(n_k):
        for s_idx in block_t_indices[b_idx]:
            sub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are okay for small intersections.
    # We will use sets for intersection in local search.
    sub_to_blocks_sets = [set(lst) for lst in sub_to_blocks]

    def local_search_improve(selected_indices, deadline):
        # selected_indices is a list of block indices
        if not selected_indices:
            return [], 0
            
        # Build coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_indices:
            coverage_count[block_t_indices[bi]] += 1
            
        current_size = len(selected_indices)
        improved = True
        no_improve_streak = 0
        rng_local = random.Random(42) # Local RNG for determinism in search steps? 
        # Using global rng is fine, but let's use a dedicated one for restarts
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # We scan the list. If a block is redundant, we remove it.
            i = 0
            while i < len(selected_indices):
                if time.time() > deadline - 2.0:
                    return selected_indices, len(selected_indices)
                    
                bi = selected_indices[i]
                ts = block_t_indices[bi]
                # Check if all covered subsets have count > 1
                if np.all(coverage_count[ts] > 1):
                    # Remove block
                    coverage_count[ts] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                    # Do not increment i, check the new element at i
                else:
                    i += 1
                    
            # Phase 2: Try swaps
            # We try 1-for-1 swaps first.
            # We iterate through blocks.
            # To avoid O(N^2) complexity, we limit the number of swap attempts or sample blocks.
            
            if time.time() > deadline - 5.0:
                break
                
            # Shuffle indices to randomize order? 
            # Let's try to find a block that can be swapped.
            # We iterate through a copy of the indices to avoid modification issues during iteration
            # But we break on first success to re-run Phase 1.
            
            # Optimization: Only attempt swaps if we have time.
            # We will iterate through blocks, but if it gets too slow, we break.
            
            # To improve efficiency, we can prioritize blocks with many "critical" subsets?
            # No, blocks with few critical subsets are easier to swap.
            
            # Let's just iterate.
            for i in range(len(selected_indices)):
                if time.time() > deadline - 2.0:
                    break
                    
                bi = selected_indices[i]
                ts_b = block_t_indices[bi]
                
                # Find critical subsets: those with count == 1
                critical_mask = (coverage_count[ts_b] == 1)
                critical_subsets = ts_b[critical_mask]
                
                if len(critical_subsets) == 0:
                    # This block is redundant, should have been removed in Phase 1.
                    coverage_count[ts_b] -= 1
                    selected_indices.pop(i)
                    current_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
                # If too many critical subsets, finding a replacement is hard.
                # Heuristic: Skip if critical sets > 5?
                # Let's try up to 5.
                if len(critical_subsets) > 5:
                    continue
                    
                # Find candidate blocks covering ALL critical subsets
                # Intersection of sub_to_blocks_sets[s] for s in critical_subsets
                # Sort critical_subsets by the size of their candidate list to optimize intersection
                cands_lists = [sub_to_blocks_sets[s] for s in critical_subsets]
                cands_lists.sort(key=len)
                
                if not cands_lists:
                    continue
                
                # Start with the smallest list
                candidates = set(cands_lists[0])
                
                for lst in cands_lists[1:]:
                    if not candidates:
                        break
                    # Intersect with next list
                    candidates = candidates.intersection(lst)
                    
                if not candidates:
                    continue
                
                # Remove the current block bi from candidates (we don't swap with itself)
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Pick a candidate
                # Heuristic: Pick the one that covers the most subsets with count > 1 (to create redundancy)
                # Or just pick one randomly from a small subset to save time.
                # Let's evaluate a few top candidates.
                cand_list = list(candidates)
                
                # If too many candidates, sample
                if len(cand_list) > 50:
                    cand_list = rng_local.sample(cand_list, 50)
                
                # Try each candidate? Or just one?
                # Trying one is faster. Trying more might find a better swap.
                # Let's try up to 5 candidates.
                found_swap = False
                for bc in cand_list:
                    # Perform swap
                    # Remove bi
                    coverage_count[ts_b] -= 1
                    selected_indices[i] = bc
                    
                    # Add bc
                    ts_bc = block_t_indices[bc]
                    coverage_count[ts_bc] += 1
                    
                    improved = True
                    found_swap = True
                    break # Restart loop to re-run Phase 1
                    
                if found_swap:
                    break
                    
        return selected_indices, len(selected_indices)

    best_solution = None
    best_size = float('inf')
    
    # Multiple Restarts with different random seeds and greedy strategies
    # We will run several restarts to diversify the initial greedy solutions.
    
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 10.0:
            break
            
        rng = random.Random(42 + r)
        
        # Greedy Set Cover Initialization
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        
        # Perturb weights slightly for diversity in restarts > 0
        if r > 0:
            # Add small random noise to weights to break ties differently
            noise = rng.random(n_t) * 0.1
            t_weights_perturbed = t_weights * (1.0 + noise)
        else:
            t_weights_perturbed = t_weights
            
        while True:
            if time.time() > deadline - 5.0:
                break
            if not uncovered.any():
                break
                
            # Calculate weighted gains
            gains = (uncovered[block_t_indices] * t_weights_perturbed[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False
            
        if not selected_blocks:
            continue
            
        # Run local search on this greedy solution
        current_indices = list(selected_blocks)
        current_indices, current_size = local_search_improve(current_indices, deadline)
        
        if current_size < best_size:
            best_size = current_size
            best_solution = [k_subsets[i] for i in current_indices]
            
        # If we have time, try to improve further with more aggressive local search
        # Or just run local search again on the current best?
        # The local search above is already iterative.
        
    if best_solution is None:
        # Fallback to a simple greedy if all restarts failed or timed out
        # This shouldn't happen if the first restart completes
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        rng = random.Random(42)
        while True:
            if time.time() > deadline - 1.0:
                break
            if not uncovered.any():
                break
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            selected_blocks.append(best_idx)
            uncovered[block_t_indices[best_idx]] = False
        if selected_blocks:
            best_solution = [k_subsets[i] for i in selected_blocks]
        else:
            best_solution = []

    return [list(b) for b in best_solution]
