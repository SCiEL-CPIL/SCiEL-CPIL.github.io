import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    import itertools
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute balls as bitsets
    # balls[w] is an integer where bit i is 1 if word i is in the ball of w
    balls = [0] * total
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        mask = 1 << w
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                mask |= (1 << new_w)
        balls[w] = mask
        
    # Verify coverage function using bitsets
    def is_covered(selected_set):
        union_mask = 0
        for c in selected_set:
            union_mask |= balls[c]
        return union_mask == (1 << total) - 1
    
    best_size = total
    best_set = None
    
    def greedy_and_optimize(random_seed=None, deadline_check=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        selected = []
        uncovered = (1 << total) - 1
        
        # Greedy phase with bitsets
        while uncovered:
            if deadline_check and time.time() > deadline_check:
                return None
            
            # Find candidate with max coverage of uncovered words
            # scores[c] = (uncovered & balls[c]).bit_count()
            # To speed this up, we might just iterate and keep track of max
            # 6561 iterations, each doing an AND and BIT_COUNT. This is very fast.
            max_score = 0
            best_candidates = []
            
            for c in range(total):
                score = (uncovered & balls[c]).bit_count()
                if score > max_score:
                    max_score = score
                    best_candidates = [c]
                elif score == max_score and score > 0:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            # Randomly pick one of the best candidates
            best_idx = best_candidates[random.randint(0, len(best_candidates) - 1)]
            
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Local Search: Removal
        # We need cover_count to check redundancy efficiently
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            # Extract bits set in balls[c] and increment cover_count
            mask = balls[c]
            while mask:
                # Get least significant set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                mask ^= lsb
        
        # Removal pass
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                # Check if any word in balls[c] has cover_count == 1
                # If min cover_count > 1, it's redundant
                mask = balls[c]
                redundant = True
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                
                if redundant:
                    # Remove c
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        mask ^= lsb
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Local Search: Swaps
        # Try to swap c with c_prime if c_prime covers all critical words of c
        # Critical words of c: words w in balls[c] where cover_count[w] == 1
        # We can precompute critical_mask[c]
        
        improved = True
        while improved and (not deadline_check or time.time() < deadline_check - 2.0):
            improved = False
            # Precompute critical masks for all selected
            critical_masks = {}
            for c in selected:
                mask = balls[c]
                crit_mask = 0
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        crit_mask |= lsb
                    mask ^= lsb
                critical_masks[c] = crit_mask
            
            # Try random swaps
            # Limit number of attempts to avoid timeout
            attempts = 0
            while attempts < 100 and (not deadline_check or time.time() < deadline_check - 2.0):
                attempts += 1
                i = random.randint(0, len(selected) - 1)
                c = selected[i]
                crit_mask = critical_masks[c]
                
                # If no critical words, we can just remove c?
                # The removal pass should have handled that.
                # But if we swapped and created redundancy, we handle it later.
                
                c_prime = random.randint(0, total - 1)
                if c_prime == c or c_prime in selected:
                    continue
                
                # Check if c_prime covers all critical words of c
                # crit_mask & ~balls[c_prime] == 0
                if crit_mask & ~balls[c_prime] == 0:
                    # Perform swap
                    # Remove c
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        mask ^= lsb
                    
                    # Add c_prime
                    mask = balls[c_prime]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        mask ^= lsb
                        
                    selected[i] = c_prime
                    improved = True
                    
                    # Run removal pass again
                    changed = True
                    while changed:
                        changed = False
                        new_selected = []
                        for cj in selected:
                            mask = balls[cj]
                            redundant = True
                            while mask:
                                lsb = mask & -mask
                                idx = lsb.bit_length() - 1
                                if cover_count[idx] == 1:
                                    redundant = False
                                    break
                                mask ^= lsb
                            
                            if redundant:
                                mask = balls[cj]
                                while mask:
                                    lsb = mask & -mask
                                    idx = lsb.bit_length() - 1
                                    cover_count[idx] -= 1
                                    mask ^= lsb
                                changed = True
                            else:
                                new_selected.append(cj)
                        selected = new_selected
                    break # Restart inner loop if improved

        if selected is not None and len(selected) > 0:
            return selected
        return None

    # Run multiple greedy+local search iterations
    num_iterations = 100
    for i in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
        
        result = greedy_and_optimize(random_seed=i, deadline_check=deadline - 5.0)
        if result is not None:
            # Verify
            if is_covered(result):
                if len(result) < best_size:
                    best_size = len(result)
                    best_set = result.copy()
                    if best_size <= 486:
                        break
            else:
                pass

    # If we found a good solution, return it
    if best_set is not None:
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

    # Fallback: Simple greedy without optimization
    selected = []
    uncovered = (1 << total) - 1
    
    while uncovered:
        max_score = 0
        best_candidates = []
        for c in range(total):
            score = (uncovered & balls[c]).bit_count()
            if score > max_score:
                max_score = score
                best_candidates = [c]
            elif score == max_score and score > 0:
                best_candidates.append(c)
        if not best_candidates:
            break
        best_idx = best_candidates[0]
        selected.append(best_idx)
        uncovered &= ~balls[best_idx]
        
    if uncovered:
        for w in range(total):
            if (uncovered >> w) & 1:
                selected.append(w)
                uncovered &= ~balls[w]

    # Quick removal pass
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        mask = balls[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask ^= lsb
            
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            mask = balls[c]
            redundant = True
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                mask ^= lsb
            
            if redundant:
                mask = balls[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    mask ^= lsb
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
