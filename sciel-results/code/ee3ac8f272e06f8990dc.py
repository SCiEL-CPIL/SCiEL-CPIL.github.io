import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard
    deadline = time.time() + 115  # 35s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    # and blocks_of_subset: for each t-subset i, list of block indices
    subsets_of_block = []
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        sub_indices = []
        for S in itertools.combinations(B, t):
            i = t_index[S]
            sub_indices.append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block.append(np.array(sub_indices, dtype=np.int32))
        
    # Convert blocks_of_subset to arrays for faster iteration? 
    # Or keep as lists if we slice them? Arrays are better for vectorized ops, but here we iterate.
    # Let's keep as lists for now, as we just need to update gains.

    best_solution = None
    best_size = float('inf')
    
    # Multi-start greedy
    # We will run several iterations.
    # To add randomness, we can shuffle the initial gains or the order of tie-breaking.
    # Or we can just run it a few times, as the greedy is deterministic unless we add noise.
    
    # Let's add a small random noise to the initial gains to break ties and explore different paths.
    # Or randomize the order of blocks considered?
    
    max_restarts = 200
    lower_bound = 131  # Known lower bound for C(19,7,4)
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        # Early termination if we've hit the lower bound
        if best_size <= lower_bound:
            break
            
        # Initialize
        uncovered_mask = np.ones(n_t, dtype=bool)
        gains = np.array([len(subs) for subs in subsets_of_block], dtype=np.float64)
        
        # Add small random noise to break symmetry/ties
        if restart > 0:
            # Scale noise based on how close we are to the best solution
            noise_scale = 0.05 + 0.05 * (1.0 - min(1.0, float(best_size - 150) / 10.0))
            gains += np.random.uniform(0, noise_scale, n_k)
            
        solution_blocks = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Find best block
            if np.max(gains) <= 0:
                break
                
            best_idx = np.argmax(gains)
            # If gain is 0, we are stuck (shouldn't happen if uncovered exists)
            if gains[best_idx] == 0:
                 # Fallback: pick any block that covers something? 
                 # If max gain is 0 but uncovered exists, it means we have a bug or logical error.
                 # Actually, if max gain is 0, it means no block covers any uncovered subset? 
                 # That's impossible if uncovered is not empty.
                 break

            solution_blocks.append(best_idx)
            
            # Update gains
            subs_b = subsets_of_block[best_idx]
            
            # Find which subsets in best_idx are actually uncovered
            # We can check uncovered_mask[subs_b]
            newly_covered_mask = uncovered_mask[subs_b]
            
            if not np.any(newly_covered_mask):
                # This block covers nothing new? Should not happen if gain > 0.
                # Gain was calculated based on uncovered.
                pass
                
            # Update gains for all blocks sharing subsets with best_idx
            # We iterate over the newly covered subsets
            for s_idx in subs_b[newly_covered_mask]:
                # Get all blocks containing s_idx
                for b_prime in blocks_of_subset[s_idx]:
                    gains[b_prime] -= 1.0
                    
            # Update uncovered_mask
            uncovered_mask[subs_b] = False
            
            # Check if all covered
            if not np.any(uncovered_mask):
                break
                
        # Solution found (or best effort)
        if len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
            # If we reached the lower bound, stop immediately
            if best_size <= lower_bound:
                break
            
    # Post-processing: Deletion Pass on best_solution (optimized)
    if best_solution is not None:
        # Build incidence matrix: rows = t_subsets, cols = blocks in solution
        n_blocks = len(best_solution)
        if n_blocks > 0:
            # Create sparse-like structure using numpy
            incidence = np.zeros((n_t, n_blocks), dtype=np.int32)
            for col, b_idx in enumerate(best_solution):
                subs_b = subsets_of_block[b_idx]
                incidence[subs_b, col] = 1
            
            cover_count = incidence.sum(axis=1)
            
            # Try to remove blocks - a block can be removed if no t-subset it covers
            # is covered only by that block (i.e., cover_count[s] == 1 and incidence[s, b] == 1)
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for col in range(len(best_solution)):
                    b_idx = best_solution[col]
                    subs_b = subsets_of_block[b_idx]
                    # Check if any subset in this block has count == 1 (must keep)
                    # Vectorized check: any subset where cover_count == 1
                    if np.any(cover_count[subs_b] == 1):
                        new_blocks.append(b_idx)
                    else:
                        # Can remove this block
                        cover_count[subs_b] -= 1
                        changed = True
                
                best_solution = new_blocks
            
    # Construct the output list of blocks
    if best_solution is None:
        # Emergency fallback: simple greedy without optimization
        # This should not be reached if the loop ran
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks

    # Return list of lists of ints
    if best_solution is None:
        # Should not reach here, but return empty list as fallback
        return []
    return [list(k_subsets[idx]) for idx in best_solution]
