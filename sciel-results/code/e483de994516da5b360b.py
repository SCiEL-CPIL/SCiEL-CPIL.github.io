"""
Optimized Ternary Covering Code K_3(6,1) solver.

Uses a hybrid approach:
1. Direct Sum initialization (tetracode × full 2D tail).
2. Aggressive local search with:
   a. Greedy removal of redundant codewords.
   b. Swap-based moves to escape local minima, guided by minimizing the number of 
      "critical" words (words covered by exactly one codeword).
   c. Multi-start with different random seeds and perturbation strategies.
"""

import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit, keep 32s buffer)

    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = 729
        powers = [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors for each word (index 0..728)
        # neighbors[i] is a list of 13 indices (i and 12 others at dist 1)
        neighbors = np.empty((total_words, 13), dtype=np.int32)
        for i in range(total_words):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1
        
        # Also precompute neighbor sets as Python sets for fast membership
        neighbor_sets = [set(neighbors[i].tolist()) for i in range(total_words)]

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
        
        def get_initial_codewords():
            initial = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    initial.append(h + list(tail))
            return initial

        def run_local_search(active_indices, inactive_indices, coverage, best_size, best_active, deadline):
            """Run local search phase, return improved (best_size, best_active)."""
            # Phase 1: Greedy Removal
            removed_any = True
            while removed_any:
                if time.time() > deadline - 3.0:
                    break
                removed_any = False
                candidates = list(active_indices)
                random.shuffle(candidates)
                
                for idx in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if idx not in active_indices:
                        continue
                    if np.min(coverage[neighbors[idx]]) >= 2:
                        active_indices.remove(idx)
                        inactive_indices.add(idx)
                        coverage[neighbors[idx]] -= 1
                        removed_any = True
                        if len(active_indices) < best_size:
                            best_size = len(active_indices)
                            best_active = set(active_indices)
                        break  # Restart removal phase with new state
            
            # Phase 2: Swap-based Local Search
            # Goal: Find swaps that maintain coverage and enable further removals
            max_swap_iterations = 50
            for swap_iter in range(max_swap_iterations):
                if time.time() > deadline - 2.0:
                    break
                
                # Compute critical words count
                critical_count = int(np.sum(coverage == 1))
                
                # Find a beneficial swap
                # For each active idx_out, find unique words it covers
                # U_out = { w in N(idx_out) | coverage[w] == 1 }
                # We need an inactive idx_in such that U_out ⊆ N(idx_in)
                
                best_swap = None
                best_swap_score = critical_count + 1  # Start with worse than current
                
                active_list = list(active_indices)
                random.shuffle(active_list)
                
                for idx_out in active_list:
                    if time.time() > deadline - 1.0:
                        break
                    
                    n_out = neighbors[idx_out]
                    unique_words = []
                    for w in n_out:
                        if coverage[w] == 1:
                            unique_words.append(w)
                    
                    if not unique_words:
                        continue
                    
                    # Try to find an inactive idx_in that covers all unique_words
                    # Sample a subset of inactive indices for speed
                    inactive_list = list(inactive_indices)
                    random.shuffle(inactive_list)
                    
                    # Check up to 200 inactive candidates
                    for idx_in in inactive_list[:200]:
                        n_in = neighbors[idx_in]
                        valid = True
                        for w in unique_words:
                            if w not in n_in:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Calculate new critical count after swap
                        # New coverage = coverage - N(idx_out) + N(idx_in)
                        # Words affected are in N(idx_out) ∪ N(idx_in)
                        
                        # Quick estimate: count changes in critical status
                        new_critical = 0
                        affected = set(n_out.tolist()) | set(n_in.tolist())
                        for w in affected:
                            new_cov = coverage[w]
                            if w in n_out:
                                new_cov -= 1
                            if w in n_in:
                                new_cov += 1
                            if new_cov == 1:
                                new_critical += 1
                        
                        # Add unchanged critical words
                        for w in range(total_words):
                            if w not in affected and coverage[w] == 1:
                                new_critical += 1
                        
                        if new_critical < best_swap_score:
                            best_swap_score = new_critical
                            best_swap = (idx_out, idx_in)
                        
                        if best_swap_score <= critical_count:
                            break  # Found a good swap, stop searching
                    
                    if best_swap and best_swap_score <= critical_count:
                        break
                
                if best_swap:
                    idx_out, idx_in = best_swap
                    active_indices.remove(idx_out)
                    inactive_indices.add(idx_out)
                    inactive_indices.remove(idx_in)
                    active_indices.add(idx_in)
                    
                    coverage[neighbors[idx_out]] -= 1
                    coverage[neighbors[idx_in]] += 1
                    
                    # Try removals again after swap
                    removed_any = True
                    while removed_any:
                        if time.time() > deadline - 1.0:
                            break
                        removed_any = False
                        candidates = list(active_indices)
                        random.shuffle(candidates)
                        for idx in candidates:
                            if time.time() > deadline - 1.0:
                                break
                            if idx not in active_indices:
                                continue
                            if np.min(coverage[neighbors[idx]]) >= 2:
                                active_indices.remove(idx)
                                inactive_indices.add(idx)
                                coverage[neighbors[idx]] -= 1
                                removed_any = True
                                if len(active_indices) < best_size:
                                    best_size = len(active_indices)
                                    best_active = set(active_indices)
                                break
                else:
                    # No valid swap found, try perturbation
                    if time.time() > deadline - 1.0:
                        break
                    # Random perturbation: swap a random active with random inactive
                    # even if it temporarily increases critical count
                    active_list = list(active_indices)
                    inactive_list = list(inactive_indices)
                    if active_list and inactive_list:
                        idx_out = random.choice(active_list)
                        idx_in = random.choice(inactive_list)
                        active_indices.remove(idx_out)
                        inactive_indices.add(idx_out)
                        inactive_indices.remove(idx_in)
                        active_indices.add(idx_in)
                        
                        coverage[neighbors[idx_out]] -= 1
                        coverage[neighbors[idx_in]] += 1
                        
                        # Try removals after perturbation
                        removed_any = True
                        while removed_any:
                            if time.time() > deadline - 1.0:
                                break
                            removed_any = False
                            candidates = list(active_indices)
                            random.shuffle(candidates)
                            for idx in candidates:
                                if time.time() > deadline - 1.0:
                                    break
                                if idx not in active_indices:
                                    continue
                                if np.min(coverage[neighbors[idx]]) >= 2:
                                    active_indices.remove(idx)
                                    inactive_indices.add(idx)
                                    coverage[neighbors[idx]] -= 1
                                    removed_any = True
                                    if len(active_indices) < best_size:
                                        best_size = len(active_indices)
                                        best_active = set(active_indices)
                                    break
        
        # Multi-start local search
        best_size = total_words
        best_active = set()
        
        # Run multiple iterations with different random seeds
        num_starts = 5
        for seed in range(num_starts):
            if time.time() > deadline - 5.0:
                break
            
            random.seed(seed * 12345 + 67890)
            
            # Get initial codewords
            initial_codewords = get_initial_codewords()
            active_indices = set(codeword_to_index(cw) for cw in initial_codewords)
            inactive_indices = set(range(total_words)) - active_indices
            
            # Compute initial coverage
            coverage = np.zeros(total_words, dtype=np.int32)
            for idx in active_indices:
                coverage[neighbors[idx]] += 1
            
            # Run local search
            best_size, best_active = run_local_search(
                active_indices, inactive_indices, coverage, 
                best_size, best_active, deadline
            )
            
            # If we reached 73 or better, stop early
            if best_size <= 73:
                break
        
        # Convert best_active to codewords
        final_codewords = [index_to_codeword(idx) for idx in best_active]
        
        # Verify coverage (quick check)
        if time.time() < deadline - 1.0:
            verify_coverage = np.zeros(total_words, dtype=np.int32)
            for cw in final_codewords:
                idx = codeword_to_index(cw)
                verify_coverage[neighbors[idx]] += 1
            if np.any(verify_coverage == 0):
                # Fallback to direct sum if verification fails
                initial_codewords = get_initial_codewords()
                return initial_codewords
        
        return final_codewords

    # General case fallback
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]
        
    return get_direct_sum()
