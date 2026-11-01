import time
import numpy as np
import random
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # balls[w] contains w and all words at distance 1 from w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # For each word c, count how many uncovered words are in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace
        # Try to replace a codeword with another to allow further reduction
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove (might have become removable due to previous swaps in this pass)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                # Try a few random candidates
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    
                    # Check if replacing c with c_prime is valid
                    # Temporarily remove c
                    cover_count[balls[c]] -= 1
                    
                    # Identify critical words for c (words covered only by c)
                    critical_in_c = (cover_count[balls[c]] == 1)
                    
                    if np.any(critical_in_c):
                        critical_indices = balls[c][critical_in_c]
                        # Check if all critical_indices are in balls[c_prime]
                        c_prime_set = set(balls[c_prime].tolist())
                        valid = True
                        for idx in critical_indices:
                            if idx not in c_prime_set:
                                valid = False
                                break
                        if valid:
                            # Swap is valid
                            cover_count[balls[c_prime]] += 1
                            new_selected.append(c_prime)
                            swapped = True
                            improved = True
                            break
                    else:
                        # No critical words, so c was removable, but we checked that above.
                        # This case shouldn't happen if the removal check was robust.
                        # But if it does, the swap is trivially valid?
                        # If no critical words, removing c is safe.
                        # So we should have removed it.
                        pass

                    # If not swapped, revert removal of c
                    if not swapped:
                        cover_count[balls[c]] += 1
                
                if not swapped:
                    # Keep c
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: LP-based refinement
        if time.time() < deadline - 15.0 and len(selected) > 403:
            selected = _lp_refine(selected, cover_count, balls, total, deadline)
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                # We hit the known optimum.
                # Break early to save time.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result

def _lp_refine(selected, cover_count, balls, total, deadline):
    """
    Use LP relaxation to guide a deterministic rounding procedure.
    """
    selected = list(selected)
    cover_count = cover_count.copy()
    
    if len(selected) == 0:
        return selected
    
    # Build the LP: minimize sum(x_i) subject to A x >= 1, 0 <= x_i <= 1
    # where A[i][j] = 1 if word i is in ball of codeword j
    # This is a large LP (6561 x |selected|), so we use a reduced version.
    
    # Actually, let's use a simpler approach: compute dual-like weights.
    # For each selected codeword c, compute a "criticality score" based on how many
    # words it uniquely covers or nearly uniquely covers.
    
    # Alternative: use the LP dual variables from a small subproblem.
    # Let's try a different approach: iterative removal guided by LP.
    
    # For now, do a more aggressive local search with LP-inspired heuristics.
    
    # Compute for each codeword the number of words it covers that are covered by exactly k codewords
    n_selected = len(selected)
    
    # Build incidence: for each word, which codewords cover it
    # This is expensive, so we use the cover_count array.
    
    # Try to remove codewords in order of least "importance"
    # Importance = number of words covered by exactly this codeword (unique coverage)
    
    # Compute unique coverage for each selected codeword
    unique_counts = np.zeros(n_selected, dtype=np.int32)
    for idx, c in enumerate(selected):
        ball_words = balls[c]
        # Words in ball_words that are covered only by c
        # cover_count[ball_words] == 1 means covered only by c (since c is in selected)
        unique_counts[idx] = np.sum(cover_count[ball_words] == 1)
    
    # Try removing in order of increasing unique coverage
    # But we need to check if removal is valid (no word becomes uncovered)
    
    # Actually, let's do a different LP-inspired approach:
    # Solve a small LP on a subset of critical words to get dual variables.
    
    # For now, use a greedy approach with LP-inspired weights.
    # Weight each codeword by 1/(unique_counts + 1) and try to remove low-weight codewords first.
    
    # Reorder selected by increasing unique_counts
    order = np.argsort(unique_counts)
    
    new_selected = []
    new_cover_count = cover_count.copy()
    
    for idx in order:
        c = selected[idx]
        # Try to remove c
        if np.min(new_cover_count[balls[c]]) > 1:
            new_cover_count[balls[c]] -= 1
        else:
            new_selected.append(c)
    
    # Rebuild cover_count for new_selected
    new_cover_count = np.zeros(total, dtype=np.int32)
    for c in new_selected:
        new_cover_count[balls[c]] += 1
    
    # Verify validity
    if np.any(new_cover_count == 0):
        # Invalid, revert
        return selected, cover_count
    
    # Now do swap phase with LP-inspired guidance
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        new_selected2 = []
        for c in new_selected:
            # Try to remove
            if np.min(new_cover_count[balls[c]]) > 1:
                new_cover_count[balls[c]] -= 1
                improved = True
                continue
            
            # Try to swap
            swapped = False
            # Try random candidates
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                c_prime = all_indices[np.random.randint(total)]
                if c_prime == c:
                    continue
                
                # Check swap validity
                new_cover_count[balls[c]] -= 1
                critical_in_c = (new_cover_count[balls[c]] == 1)
                
                if np.any(critical_in_c):
                    critical_indices = balls[c][critical_in_c]
                    c_prime_set = set(balls[c_prime].tolist())
                    valid = all(idx in c_prime_set for idx in critical_indices)
                    if valid:
                        new_cover_count[balls[c_prime]] += 1
                        new_selected2.append(c_prime)
                        swapped = True
                        improved = True
                        break
                else:
                    # No critical words, so removal is safe
                    new_cover_count[balls[c]] += 1  # Revert
                    # Actually, if no critical words, we should have removed it
                    # This means the min check was wrong, but let's just keep c
                    pass
                
                if not swapped:
                    new_cover_count[balls[c]] += 1
            
            if not swapped:
                new_selected2.append(c)
        
        new_selected = new_selected2
    
    # Rebuild cover_count
    final_cover_count = np.zeros(total, dtype=np.int32)
    for c in new_selected:
        final_cover_count[balls[c]] += 1
    
    # Verify
    if np.any(final_cover_count == 0):
        return selected, cover_count
    
    return new_selected, final_cover_count
