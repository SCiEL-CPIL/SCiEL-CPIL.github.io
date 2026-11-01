import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Construct balls matrix once outside the loop
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

    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    # Since balls is precomputed, we can run more iterations or let it run until deadline
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen
                
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
            continue # Invalid solution
            
        # 2. Local Search / Pruning
        # Try to remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # 3. Local Search: Swap/Move
        # Try to replace a codeword with another to enable further reduction
        # We do a few passes of swapping
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Pick a random codeword to try replacing
            if not selected:
                break
            c_idx = np.random.randint(len(selected))
            c_val = selected[c_idx]
            
            # Pick a random replacement
            r_val = np.random.randint(total)
            if r_val == c_val:
                continue
                
            # Check if replacing c_val with r_val maintains coverage
            # This requires checking if any word covered ONLY by c_val is NOT covered by r_val
            # i.e., min(cover_count[balls[c_val]]) > 1 OR (min == 1 AND that word is covered by r_val)
            # Simpler: 
            # 1. Remove c_val (tentatively)
            # 2. Add r_val (tentatively)
            # 3. Check if any count becomes 0
            # 4. If yes, revert. If no, keep and try to remove redundancies.
            
            # To do this efficiently:
            # The words that might become uncovered are those covered by c_val but not by r_val.
            # Let C_c = balls[c_val], C_r = balls[r_val]
            # Words in C_c \ C_r are at risk.
            # We need to ensure that for all w in C_c \ C_r, cover_count[w] > 1 (so removing c_val doesn't drop it to 0) 
            # OR cover_count[w] == 1 (so it drops to 0, which is bad UNLESS r_val covers it, but r_val doesn't cover it by definition of the set difference).
            # So we need: for all w in C_c \ C_r, cover_count[w] > 1.
            
            # Efficient check:
            # min_count_at_risk = min(cover_count[w] for w in C_c if w not in C_r)
            # If min_count_at_risk > 1, the swap is safe.
            
            # Using numpy:
            # mask_c = np.zeros(total, dtype=bool); mask_c[balls[c_val]] = True
            # mask_r = np.zeros(total, dtype=bool); mask_r[balls[r_val]] = True
            # risk_mask = mask_c & ~mask_r
            # if np.any(risk_mask):
            #     if np.min(cover_count[risk_mask]) > 1:
            #         # Safe to swap
            #         pass
            #     else:
            #         # Unsafe
            #         continue
            # else:
            #     # Safe to swap (no risk words)
            #     pass
            
            # This is O(N) per swap attempt. With 6561 words, it's fast.
            # But creating masks is slow.
            # Alternative: 
            # risk_words = np.setdiff1d(balls[c_val], balls[r_val])
            # if len(risk_words) > 0:
            #     if np.min(cover_count[risk_words]) > 1:
            #         # Safe
            #         pass
            #     else:
            #         continue
            # else:
            #     # Safe
            
            risk_words = np.setdiff1d(balls[c_val], balls[r_val])
            safe = True
            if len(risk_words) > 0:
                if np.min(cover_count[risk_words]) <= 1:
                    safe = False
            
            if safe:
                # Perform swap
                cover_count[balls[c_val]] -= 1
                cover_count[balls[r_val]] += 1
                selected[c_idx] = r_val
                improved = True
                
                # Try to remove redundancies after swap
                changed = True
                while changed and time.time() < deadline - 5.0:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            changed = True
                        else:
                            new_selected.append(c)
                    selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

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
