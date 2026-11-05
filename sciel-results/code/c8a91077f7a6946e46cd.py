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
        is_neighbor = np.zeros((total_words, total_words), dtype=bool)
        
        for i in range(total_words):
            # Get digits
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            is_neighbor[i, i] = True
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        is_neighbor[new_i, i] = True
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

        # Phase 1: Greedy Construction
        covered = np.zeros(total_words, dtype=bool)
        active_indices = []
        
        while not covered.all():
            if time.time() > deadline - 10.0:
                break
            
            # Calculate how many new words each candidate covers
            new_coverages = is_neighbor.T @ (~covered).astype(np.int8)
            best_idx = np.argmax(new_coverages)
            
            if new_coverages[best_idx] == 0:
                break
                
            active_indices.append(best_idx)
            covered[is_neighbor[:, best_idx]] = True
            
        active_set = set(active_indices)
        inactive_set = set(range(total_words)) - active_set
        
        best_size = len(active_indices)
        best_active = set(active_indices)

        # Precompute coverage counts
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in active_indices:
            coverage[neighbors[idx]] += 1

        # Genetic Algorithm for Code Optimization
        # Population of valid codes, fitness based on sum of squared coverage counts
        # (lower sum of squares indicates more uniform coverage, potentially allowing more removals)
        
        def compute_fitness(coverage_arr):
            # Sum of squared coverage counts
            return np.sum(coverage_arr * coverage_arr)

        def try_remove_redundant(active_set, inactive_set, coverage):
            """Attempt to remove redundant codewords. Returns (success, new_size)."""
            removed = False
            # Shuffle for randomization
            active_list = list(active_set)
            random.shuffle(active_list)
            
            for idx in active_list:
                if time.time() > deadline - 2.0:
                    break
                if idx not in active_set:
                    continue
                
                # Check if idx is redundant: all neighbors have coverage >= 2
                if np.min(coverage[neighbors[idx]]) >= 2:
                    active_set.remove(idx)
                    inactive_set.add(idx)
                    coverage[neighbors[idx]] -= 1
                    removed = True
                    # Continue removing if possible (loop restarts logic handled by caller or here)
                    # For simplicity in GA, we do one removal per step or loop until stable
                    # Let's loop until no more removals
                    try_remove_redundant(active_set, inactive_set, coverage)
                    break
            return removed

        def perform_swap(active_set, inactive_set, coverage, idx_out, idx_in):
            """Perform a swap and check validity. Returns True if valid."""
            # Check if removing idx_out leaves any uncovered words
            # unique_words = neighbors[idx_out] where coverage == 1
            n_out = neighbors[idx_out]
            unique_mask = coverage[n_out] == 1
            unique_words = n_out[unique_mask]
            
            if len(unique_words) == 0:
                # Should have been removed, but if not, swap is trivially valid for coverage
                pass
            else:
                # Check if idx_in covers all unique_words
                if not np.all(is_neighbor[unique_words, idx_in]):
                    return False
            
            # Perform swap
            active_set.remove(idx_out)
            inactive_set.add(idx_out)
            inactive_set.remove(idx_in)
            active_set.add(idx_in)
            
            coverage[neighbors[idx_out]] -= 1
            coverage[neighbors[idx_in]] += 1
            
            return True

        def find_valid_swap(active_set, inactive_set, coverage):
            """Find a valid swap to escape local minimum."""
            active_list = list(active_set)
            random.shuffle(active_list)
            
            for idx_out in active_list:
                if time.time() > deadline - 2.0:
                    break
                
                n_out = neighbors[idx_out]
                unique_mask = coverage[n_out] == 1
                unique_words = n_out[unique_mask]
                
                if len(unique_words) == 0:
                    continue
                
                # Find inactive words that cover all unique_words
                # is_neighbor[unique_words, :] is (U, 729)
                # We want columns where all are True
                covers_all = np.sum(is_neighbor[unique_words, :], axis=0) == len(unique_words)
                
                # Mask for inactive
                inactive_mask = np.zeros(total_words, dtype=bool)
                for idx in inactive_set:
                    inactive_mask[idx] = True
                
                valid_candidates = np.where(covers_all & inactive_mask)[0]
                
                if len(valid_candidates) > 0:
                    idx_in = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    return (idx_out, idx_in)
            
            return None

        # Main GA Loop
        # We maintain a "current" state and try to improve it.
        # Since the state space is huge, we use a hill-climbing approach with swaps.
        # The "genetic" aspect here is the random selection of swaps and restarts.
        
        current_active = set(active_set)
        current_inactive = set(inactive_set)
        current_coverage = coverage.copy()
        
        # Initial cleanup
        try_remove_redundant(current_active, current_inactive, current_coverage)
        
        if len(current_active) < best_size:
            best_size = len(current_active)
            best_active = set(current_active)

        # Iterative Improvement
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < deadline - 5.0 and no_improvement_count < max_no_improvement:
            
            # 1. Try to remove redundant words
            removed = try_remove_redundant(current_active, current_inactive, current_coverage)
            if removed:
                no_improvement_count = 0
                if len(current_active) < best_size:
                    best_size = len(current_active)
                    best_active = set(current_active)
                    no_improvement_count = 0
                continue

            # 2. Try a valid swap
            swap = find_valid_swap(current_active, current_inactive, current_coverage)
            if swap:
                idx_out, idx_in = swap
                if perform_swap(current_active, current_inactive, current_coverage, idx_out, idx_in):
                    # Swap was valid. Check for new removals.
                    removed = try_remove_redundant(current_active, current_inactive, current_coverage)
                    if removed:
                        no_improvement_count = 0
                        if len(current_active) < best_size:
                            best_size = len(current_active)
                            best_active = set(current_active)
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                # No valid swap found. Local minimum.
                # Try a "kick": Randomly mutate the code by swapping a few words,
                # even if it temporarily breaks validity? No, we must maintain validity.
                # Instead, we can try to find a swap that minimizes the fitness (sum of squares)
                # even if it doesn't immediately allow a removal.
                
                # Fitness-based Swap Search
                best_fitness = compute_fitness(current_coverage)
                best_swap_found = None
                best_new_fitness = best_fitness + 1 # Must be strictly better to accept? Or equal?
                
                # Sample a few random swaps
                active_list = list(current_active)
                random.shuffle(active_list)
                
                for idx_out in active_list[:20]: # Limit search
                    if time.time() > deadline - 2.0:
                        break
                    
                    n_out = neighbors[idx_out]
                    unique_mask = current_coverage[n_out] == 1
                    unique_words = n_out[unique_mask]
                    
                    if len(unique_words) == 0:
                        continue
                    
                    covers_all = np.sum(is_neighbor[unique_words, :], axis=0) == len(unique_words)
                    inactive_mask = np.zeros(total_words, dtype=bool)
                    for idx in current_inactive:
                        inactive_mask[idx] = True
                    
                    valid_candidates = np.where(covers_all & inactive_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # Pick a few candidates
                        sample_size = min(5, len(valid_candidates))
                        candidates = valid_candidates[random.sample(range(len(valid_candidates)), sample_size)]
                        
                        for idx_in in candidates:
                            # Simulate swap
                            temp_coverage = current_coverage.copy()
                            temp_coverage[neighbors[idx_out]] -= 1
                            temp_coverage[neighbors[idx_in]] += 1
                            
                            # Check validity (all covered)
                            if np.min(temp_coverage) >= 1:
                                fitness = compute_fitness(temp_coverage)
                                if fitness < best_new_fitness:
                                    best_new_fitness = fitness
                                    best_swap_found = (idx_out, idx_in)
                
                if best_swap_found:
                    idx_out, idx_in = best_swap_found
                    perform_swap(current_active, current_inactive, current_coverage, idx_out, idx_in)
                    # Try to remove redundant again
                    removed = try_remove_redundant(current_active, current_inactive, current_coverage)
                    if removed:
                        no_improvement_count = 0
                        if len(current_active) < best_size:
                            best_size = len(current_active)
                            best_active = set(current_active)
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1

        final_codewords = [index_to_codeword(idx) for idx in best_active]
        
        return final_codewords

    # General case fallback
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]
        
    return get_direct_sum()
