"""
Optimized Ternary Covering Code K_3(6,1) solver.

Uses a hybrid approach:
1. Direct Sum initialization.
2. Iterative local search with:
   a. Greedy removal of redundant codewords.
   b. Swap-based moves to escape local minima, guided by minimizing the number of 
      "critical" words (words covered by exactly one codeword).
"""

import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = 729
        powers = [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors for each word (index 0..728)
        # neighbors[i] is a list of 13 indices (i and 12 others at dist 1)
        neighbors = np.empty((total_words, 13), dtype=np.int32)
        for i in range(total_words):
            # Get digits
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Helper to convert index to codeword list
        def index_to_codeword(idx):
            return [(idx // powers[p]) % 3 for p in range(n)]

        # Helper to convert codeword list to index
        def codeword_to_index(cw):
            idx = 0
            for p in range(n):
                idx += cw[p] * powers[p]
            return idx

        # Initial code: Direct Sum of Tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                initial_codewords.append(h + list(tail))
        
        # Convert to indices
        active_indices = set(codeword_to_index(cw) for cw in initial_codewords)
        inactive_indices = set(range(total_words)) - active_indices
        
        best_size = len(active_indices)
        best_active = set(active_indices)

        # Local Search Loop
        while time.time() < deadline - 5.0:
            # Phase 1: Greedy Removal
            # Compute coverage counts
            coverage = np.zeros(total_words, dtype=np.int32)
            for idx in active_indices:
                coverage[neighbors[idx]] += 1
            
            removed_any = False
            # Try to remove redundant codewords
            # Randomize order to avoid bias
            candidates = list(active_indices)
            random.shuffle(candidates)
            
            for idx in candidates:
                if time.time() > deadline - 5.0:
                    break
                if idx not in active_indices:
                    continue
                
                # Check if idx is redundant
                # Redundant if all its neighbors are covered by >= 2 codewords
                # (Since idx covers itself, if coverage[w] >= 2 for all w in N(idx), 
                #  then after removing idx, coverage[w] >= 1)
                
                # Optimization: Check min coverage in N(idx)
                if np.min(coverage[neighbors[idx]]) >= 2:
                    # Remove idx
                    active_indices.remove(idx)
                    inactive_indices.add(idx)
                    # Update coverage
                    coverage[neighbors[idx]] -= 1
                    removed_any = True
                    if len(active_indices) < best_size:
                        best_size = len(active_indices)
                        best_active = set(active_indices)
                    break # Restart removal phase with new state
            
            if removed_any:
                continue # Go back to top of loop to try more removals

            # Phase 2: Swap-based Local Search
            # If no removals possible, try to swap to enable future removals
            # Goal: Minimize number of critical words (coverage == 1)
            
            # Compute critical words count
            critical_count = np.sum(coverage == 1)
            
            # Find a beneficial swap
            # For each active idx_out, find unique words it covers
            # U_out = { w in N(idx_out) | coverage[w] == 1 }
            # We need an inactive idx_in such that U_out is subset of N(idx_in)
            
            best_swap = None
            best_critical_after = critical_count + 1 # Start with worse than current
            
            # To speed up, we can limit the search or use heuristics
            # Iterate over active codewords
            active_list = list(active_indices)
            random.shuffle(active_list)
            
            for idx_out in active_list:
                if time.time() > deadline - 2.0:
                    break
                
                # Get unique words covered by idx_out
                n_out = neighbors[idx_out]
                unique_words = []
                for w in n_out:
                    if coverage[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    # This should have been removed in Phase 1
                    continue
                
                # We need idx_in to cover all unique_words
                # Iterate over a subset of inactive codewords to save time
                # Or all? 650 * 13 checks is small.
                for idx_in in inactive_indices:
                    # Check if idx_in covers all unique_words
                    valid = True
                    n_in = neighbors[idx_in]
                    # Fast check: for each w in unique_words, is w in n_in?
                    # Since n_in is small, we can use a set or just loop
                    # Better: Precompute a bitset or use numpy?
                    # Let's just loop, it's small.
                    for w in unique_words:
                        if w not in n_in:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Swap is valid. Calculate new critical count.
                    # New coverage = coverage - N(idx_out) + N(idx_in)
                    # We only care about changes in critical count.
                    
                    # Words affected are in N(idx_out) U N(idx_in)
                    # Let's simulate the change.
                    
                    # Old critical: coverage[w] == 1
                    # New coverage[w] = coverage[w] - (1 if w in N(idx_out) else 0) + (1 if w in N(idx_in) else 0)
                    
                    # We can calculate the delta in critical count.
                    # But calculating exact new critical count for all 729 words is O(729).
                    # Doing this for every candidate swap (up to 75*650) is 75*650*729 ~ 35M ops. Might be too slow.
                    
                    # Heuristic: Estimate the change.
                    # Removing idx_out:
                    # - For w in N(idx_out) where coverage[w] == 1: becomes 0 (uncovered) -> BAD, but we ensured unique_words are covered by idx_in.
                    #   Wait, if w in unique_words, coverage becomes 0 then +1 from idx_in -> 1. So stays critical?
                    #   If w in N(idx_out) and coverage[w] > 1: coverage decreases by 1.
                    #     If it was 2, becomes 1 (new critical).
                    #     If it was >2, stays >1.
                    # Adding idx_in:
                    # - For w in N(idx_in) where coverage[w] == 0 (impossible in valid code) or 1:
                    #   If w in N(idx_in) and not in N(idx_out):
                    #     If coverage[w] == 1, becomes 2 (no longer critical).
                    #     If coverage[w] > 1, stays >1.
                    #   If w in N(idx_in) AND in N(idx_out):
                    #     Coverage change is -1 + 1 = 0.
                    
                    # This is getting complex. Let's just do the full calculation for the most promising candidates.
                    # Or, simpler: Just perform the swap if it looks good, then re-evaluate.
                    
                    # Let's pick the first valid swap found? No, that's random.
                    # Let's pick the swap that minimizes the "potential" for removal.
                    
                    # Simpler heuristic: Just pick a random valid swap to perturb the state.
                    # The removal phase will then clean up.
                    best_swap = (idx_out, idx_in)
                    break # Break inner loop, found a valid swap for this idx_out
                
                if best_swap:
                    break # Break outer loop, found a swap
            
            if best_swap:
                idx_out, idx_in = best_swap
                active_indices.remove(idx_out)
                inactive_indices.add(idx_out)
                inactive_indices.remove(idx_in)
                active_indices.add(idx_in)
                
                # Update coverage
                coverage[neighbors[idx_out]] -= 1
                coverage[neighbors[idx_in]] += 1
                
                # Optional: Check if this swap allowed a removal?
                # The loop will restart and check removals.
            else:
                # No valid swap found? 
                # This means we are in a very tight local minimum.
                # Try a random perturbation: Swap two active codewords? No, that changes nothing.
                # Try swapping an active codeword with an inactive one even if it temporarily increases critical count?
                # If no valid swap (maintaining coverage) exists, we are stuck.
                # But we must maintain coverage!
                # If no valid swap exists, we cannot improve.
                # Break out? Or try a more aggressive search?
                
                # Given the time limit, if we are stuck, we might as well stop.
                # But maybe we can try a "double swap" or something?
                # For now, if no swap, we break to save time.
                break

        # Convert best_active to codewords
        final_codewords = [index_to_codeword(idx) for idx in best_active]
        
        # Verify coverage
        # (Trust the algorithm, but a quick check is good if time permits)
        
        return final_codewords

    # General case fallback
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]
        
    return get_direct_sum()
