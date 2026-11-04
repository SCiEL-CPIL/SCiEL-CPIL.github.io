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
    # word_to_idx: dict or array. Since max idx is 6560, we can use an array or just compute index directly.
    # Let's create a list of all words as tuples for easy conversion, but we need to map word -> index.
    # Actually, we can just iterate 0 to total_points-1 and decode to get the word, or encode the word to get index.
    
    # Pre-compute the coverage sets for every possible codeword index.
    # coverage[i] will be an array of 17 indices (the word itself + 16 neighbors).
    
    # Helper to convert index (0..6560) to list of digits (base 3, length 8)
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Helper to get neighbors of a word (index)
    # We need to generate the 16 neighbors for each of the 6561 words.
    # This is 6561 * 16 operations.
    
    coverage_sets = np.empty((total_points, 1 + n * (q - 1)), dtype=np.int32)
    
    # Precompute powers of 3 for index calculation
    # index = d0 + d1*3 + d2*3^2 ...
    # To change digit i by delta, we add delta * 3^i.
    # We must handle wrap-around: if digit is 0 and delta is -1, it becomes 2 and we subtract 2*3^i.
    # If digit is 2 and delta is +1, it becomes 0 and we subtract 2*3^i.
    
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
            # Calculate index change
            if new_digit == 0:
                # Wrapped from 2 to 0: subtract 2 * power
                delta_idx = -2 * power
            else:
                delta_idx = power
            coverage_sets[idx, neighbor_idx] = idx + delta_idx
            neighbor_idx += 1
            
            # Try subtracting 1
            new_digit = (original_digit - 1) % q
            if new_digit == 2:
                # Wrapped from 0 to 2: add 2 * power
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
            
        # Calculate how many uncovered points each candidate covers
        # is_covered[coverage_sets] gives a boolean matrix of shape (6561, 17)
        # ~is_covered[coverage_sets] gives True for uncovered points
        # sum along axis 1 gives the count of uncovered points for each candidate
        
        # This is the heavy operation, but vectorized.
        uncovered_counts = np.sum(~is_covered[coverage_sets], axis=1)
        
        best_idx = np.argmax(uncovered_counts)
        best_count = uncovered_counts[best_idx]
        
        if best_count == 0:
            break
            
        # Select this codeword
        selected_codewords.append(best_idx)
        
        # Mark its covered points as covered
        is_covered[coverage_sets[best_idx]] = True
        
        # Check if all covered
        if np.all(is_covered):
            break
            
        # Optional: early exit if we have too many, though greedy should be fine
        # If len(selected_codewords) > 1000: break # Just in case

    # 3. Pruning pass: remove redundant codewords.
    # A selected codeword c is redundant if every point in its coverage ball
    # is already covered by the union of the other selected codewords' balls.
    # We iterate removal in order of least "unique" contribution.
    if len(selected_codewords) > 1:
        # Build a boolean matrix: rows = selected codewords, cols = points in their ball
        # But simpler: for each selected codeword, track which points ONLY it covers.
        # We maintain a count array: for each point, how many selected codewords cover it.
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_codewords:
            coverage_count[coverage_sets[idx]] += 1

        # A point is "uniquely covered" by codeword c if coverage_count[point] == 1
        # and c covers it. But coverage_count counts all selected codewords.
        # To check redundancy of codeword c: remove it, then check if any point
        # in its ball has coverage_count == 1 (i.e., only c covered it).
        # Equivalently: c is redundant iff no point in its ball has coverage_count == 1.

        # We'll iterate: find a redundant codeword, remove it, update counts, repeat.
        # To be efficient, we track for each selected codeword how many points it uniquely covers.
        # unique_count[i] = number of points p in ball(c_i) such that coverage_count[p] == 1
        # If unique_count[i] == 0, codeword i is redundant.

        # Initialize unique counts
        unique_counts = np.zeros(len(selected_codewords), dtype=np.int32)
        for i, idx in enumerate(selected_codewords):
            ball = coverage_sets[idx]
            unique_counts[i] = np.sum(coverage_count[ball] == 1)

        # Iteratively remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Find any codeword with unique_count == 0
            redundant_indices = np.where(unique_counts == 0)[0]
            if len(redundant_indices) == 0:
                break
            # Remove the first one (arbitrary choice; could pick lowest index for determinism)
            rem_idx = redundant_indices[0]
            rem_codeword = selected_codewords[rem_idx]
            ball = coverage_sets[rem_codeword]

            # Decrement coverage_count for points in this ball
            coverage_count[ball] -= 1

            # Update unique_counts for other codewords:
            # For each other codeword j, its ball may overlap with rem ball.
            # A point p in ball_j had coverage_count[p] before decrement.
            # After decrement, coverage_count[p] is one less.
            # unique_count[j] = # points in ball_j with coverage_count == 1 (after update).
            # We need to recompute or update incrementally.
            # Incremental: for points in intersection of ball_j and ball_rem:
            #   before: cov_old, after: cov_old - 1
            #   if cov_old == 1: point was unique to j (if j covered it), now cov=0 -> but wait,
            #   if cov_old == 1 and j covers p, then j was the only cover, so removing rem shouldn't
            #   affect j's uniqueness for p... actually if cov_old == 1, only one codeword covers p.
            #   If that codeword is j, then rem doesn't cover p (since rem is being removed and
            #   cov_old counts rem too). Contradiction: if rem covers p, cov_old >= 2 if j also covers.
            #   So if cov_old == 1, either only j covers (rem doesn't cover p) or only rem covers (j doesn't).
            #   So for points in ball_j ∩ ball_rem, cov_old >= 2.
            #   After decrement, cov_new = cov_old - 1 >= 1.
            #   If cov_old == 2, cov_new == 1: point becomes uniquely covered by j (if j covers it and
            #   no other remaining codeword covers it). But cov_old == 2 means exactly two codewords covered:
            #   j and rem (since we're removing rem). So after removal, only j covers -> unique to j.
            #   So unique_count[j] increases by 1 for each such point.
            #   If cov_old > 2, cov_new > 1, no change to uniqueness.
            #   If cov_old == 1: impossible for points in both balls as argued.
            #
            # For points in ball_j \ ball_rem: no change.
            # For points in ball_rem \ ball_j: not in ball_j, irrelevant.
            #
            # So: unique_count[j] += (# points in ball_j ∩ ball_rem with cov_old == 2)
            # But we already decremented coverage_count, so cov_old == 2 means coverage_count[p] == 1 now.
            # So: unique_count[j] += # points in ball_j ∩ ball_rem with coverage_count[p] == 1 (after decrement).
            #
            # Also, for the removed codeword, we don't care.
            #
            # This is still O(num_selected * 17) per removal if we compute intersections.
            # Given ~486 codewords and up to ~486 removals, worst case ~486*486*17 ≈ 4M ops, fine.

            # For efficiency, iterate over other codewords and compute intersection count.
            # But we can vectorize: for each j, check ball_j ∩ ball_rem where coverage_count == 1.
            # Let's just recompute unique_counts for all remaining codewords from scratch each iteration?
            # That's O(num_selected * 17) per iteration, and num_selected decreases.
            # Total: sum_{k=1}^{num_removed} (num_selected - k + 1) * 17 ≈ 17 * num_removed * avg_selected
            # With ~100 removals and ~400 avg, that's 17*100*400 = 680K, fine.

            # Remove from selected_codewords
            selected_codewords.pop(rem_idx)

            # Recompute unique_counts for all remaining
            for i, idx in enumerate(selected_codewords):
                ball = coverage_sets[idx]
                unique_counts[i] = np.sum(coverage_count[ball] == 1)
            unique_counts = unique_counts[:len(selected_codewords)]  # trim
            changed = True

    # Convert indices back to list of lists of integers
    result = []
    for idx in selected_codewords:
        result.append(idx_to_word(idx))

    return result

# EVOLVE-BLOCK-END
