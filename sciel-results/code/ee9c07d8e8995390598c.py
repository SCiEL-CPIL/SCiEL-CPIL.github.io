"""Rewritten program using Greedy Set Cover with NumPy for ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We implement a general greedy solver for these parameters, 
    # but optimize heavily for q=3, n=8.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple construction if parameters differ, 
        # though the task contract implies specific values.
        # For safety, we return a valid (though potentially suboptimal) code.
        # The direct sum method works for n>=4.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    # Specific optimization for q=3, n=8, radius=1
    total_points = q ** n  # 6561
    
    # 1. Generate all possible codewords and map them to indices
    # We represent each word as an integer in base 3.
    
    # Helper to convert index (0..6560) to list of digits (base 3, length 8)
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute the coverage sets for every possible codeword index.
    # coverage_sets[i] will be an array of 17 indices (the word itself + 16 neighbors).
    
    coverage_sets = np.empty((total_points, 1 + n * (q - 1)), dtype=np.int32)
    
    # Precompute powers of 3 for index calculation
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for idx in range(total_points):
        word = idx_to_word(idx)
        coverage_sets[idx, 0] = idx
        
        neighbor_idx = 1
        for pos in range(n):
            original_digit = word[pos]
            power = powers[pos]
            
            # Try adding 1
            new_digit = (original_digit + 1) % q
            if new_digit == 0:
                delta_idx = -2 * power
            else:
                delta_idx = power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1
            
            # Try subtracting 1
            new_digit = (original_digit - 1) % q
            if new_digit == 2:
                delta_idx = 2 * power
            else:
                delta_idx = -power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1

    # 2. Greedy Set Cover
    is_covered = np.zeros(total_points, dtype=bool)
    selected_codewords = []
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    while True:
        if time.time() > deadline:
            break
            
        uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)
        best_idx = np.argmax(uncovered_counts)
        best_count = uncovered_counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_codewords.append(best_idx)
        is_covered[coverage_sets[best_idx]] = True
        
        if np.all(is_covered):
            break

    # 3. Pruning and Local Search
    # We maintain coverage_count: number of selected codewords covering each point.
    coverage_count = np.zeros(total_points, dtype=np.int32)
    for idx in selected_codewords:
        coverage_count[coverage_sets[idx]] += 1
        
    selected_set = set(selected_codewords)

    def try_prune():
        """Remove redundant codewords. Returns True if any removed."""
        nonlocal coverage_count, selected_set, selected_codewords
        removed_any = False
        # List to avoid modification during iteration
        to_remove = []
        for c in selected_codewords:
            ball = coverage_sets[c]
            # A codeword is redundant if no point in its ball is covered ONLY by it.
            # i.e., coverage_count[ball] > 1 for all points in ball.
            if not np.any(coverage_count[ball] == 1):
                to_remove.append(c)
                
        for c in to_remove:
            ball = coverage_sets[c]
            coverage_count[ball] -= 1
            selected_set.remove(c)
            selected_codewords.remove(c)
            removed_any = True
            
        return removed_any

    # Initial Prune
    try_prune()

    # Local Search: Swap and Reduce
    # We try to swap a codeword c with c' such that c' covers all unique points of c,
    # and then we can remove another codeword.
    
    while time.time() < deadline:
        # Prune first
        if not try_prune():
            # No simple removals, try swaps
            swapped = False
            # Iterate over a copy of the list to allow modification
            # We need to find c in selected such that swapping c -> c' allows a removal.
            # To be efficient, we check each c in selected.
            
            # We sort candidates by potential? No, just iterate.
            # To avoid getting stuck in a cycle, we can track recent swaps, but for now, simple iteration.
            
            for c in selected_codewords[:]: # Copy
                ball_c = coverage_sets[c]
                # Find unique points of c
                unique_mask = (coverage_count[ball_c] == 1)
                if not np.any(unique_mask):
                    # Should have been pruned, but handle just in case
                    continue
                
                unique_points = ball_c[unique_mask]
                
                # Find candidates c' that cover all unique_points
                # coverage_sets[:, unique_points] has shape (6561, len(unique_points))
                # We want rows where all are True.
                # This is a bit memory intensive if unique_points is large, but max 17.
                covers_unique = np.all(coverage_sets[:, unique_points], axis=1)
                
                # Get indices of candidates that cover unique points
                candidate_indices = np.where(covers_unique)[0]
                
                # Filter out those already in selected_set
                # and c itself (though c covers its own unique points, swapping c->c is useless)
                valid_candidates = [c_prime for c_prime in candidate_indices if c_prime not in selected_set]
                
                if not valid_candidates:
                    continue
                    
                # Try each valid candidate (limit to a few to save time if many)
                # We prioritize candidates that might enable a prune.
                # Heuristic: candidates with high degree (cover many points) are more likely to help.
                # But we don't have precomputed degrees.
                # Let's just try the first few.
                
                for c_prime in valid_candidates[:10]:
                    # Backup state
                    old_coverage_count = coverage_count.copy()
                    old_selected_set = selected_set.copy()
                    old_selected_list = selected_codewords.copy()
                    
                    # Perform Swap: Remove c, Add c_prime
                    coverage_count[coverage_sets[c]] -= 1
                    coverage_count[coverage_sets[c_prime]] += 1
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    selected_codewords.remove(c)
                    selected_codewords.append(c_prime)
                    
                    # Try to prune
                    if try_prune():
                        # Improvement found! Keep the swap and prune result.
                        swapped = True
                        break
                    else:
                        # No improvement, revert
                        coverage_count = old_coverage_count
                        selected_set = old_selected_set
                        selected_codewords = old_selected_list
                        
                if swapped:
                    break # Restart the outer while loop to prune again
            
            if not swapped:
                # No swaps led to improvement, stop local search
                break

    # 4. Final Prune
    try_prune()

    # Convert indices back to list of lists of integers
    result = []
    for idx in selected_codewords:
        result.append(idx_to_word(idx))

    return result

# EVOLVE-BLOCK-END
