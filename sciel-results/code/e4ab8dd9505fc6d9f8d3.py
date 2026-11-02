import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls: balls[w] is a list of 17 indices
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
    
    # Precompute balls as lists for fast iteration
    balls_list = [list(balls[w]) for w in range(total)]
    
    # Precompute bit-parallel membership using Python integers as bitmasks
    # ball_membership_bits[v] = integer where bit w is set if v is in balls[w]
    ball_membership_bits = [0] * total
    for w in range(total):
        bit = 1 << w
        for v in balls[w]:
            ball_membership_bits[v] |= bit
    
    # Helper: get list of set bits from a Python int bitmask
    def bits_to_list(mask):
        result = []
        while mask:
            # Get lowest set bit
            lb = mask & (-mask)
            result.append(lb.bit_length() - 1)
            mask ^= lb
        return result
    
    best_size = total
    best_set = None
    best_cc = None
    
    def greedy_cover():
        """Greedy set cover with randomization."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = int(candidates[np.random.randint(len(candidates))])
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        return selected, cover_count
    
    def remove_redundant(selected, cc):
        """Remove codewords whose removal doesn't uncover any word."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected
    
    def one_swap(selected, cc):
        """Try to swap each codeword with another that covers its critical words."""
        improved = True
        while improved:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cc[balls[c]] == 1)
                critical_words = balls_list[c][critical_mask]
                
                if len(critical_words) == 0:
                    cc[balls[c]] -= 1
                    improved = True
                    continue
                
                # Find candidates that cover all critical words using bit-parallel intersection
                candidate_mask = None
                for cw in critical_words:
                    bm = ball_membership_bits[cw]
                    if candidate_mask is None:
                        candidate_mask = bm
                    else:
                        candidate_mask &= bm
                
                if candidate_mask is None or candidate_mask == 0:
                    new_selected.append(c)
                    continue
                
                # Exclude c itself
                candidate_mask &= ~(1 << c)
                if candidate_mask == 0:
                    new_selected.append(c)
                    continue
                
                valid_candidates = bits_to_list(candidate_mask)
                c_prime = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                
                cc[balls[c]] -= 1
                cc[balls[c_prime]] += 1
                new_selected.append(c_prime)
                improved = True
            
            selected = new_selected
        return selected
    
    def two_exchange(selected, cc, num_trials):
        """Try 2-exchange: remove 2 codewords, add 1 that covers the gap."""
        if len(selected) < 3:
            return selected, False
        
        for _ in range(num_trials):
            idx1 = random.randint(0, len(selected) - 1)
            idx2 = random.randint(0, len(selected) - 1)
            if idx1 == idx2:
                continue
            
            c1 = selected[idx1]
            c2 = selected[idx2]
            
            # Temporarily remove
            cc[balls[c1]] -= 1
            cc[balls[c2]] -= 1
            
            uncovered_after = (cc == 0)
            
            if np.any(uncovered_after):
                uncovered_words = np.where(uncovered_after)[0]
                
                # Find c3 that covers all uncovered words
                candidate_mask = None
                for uw in uncovered_words:
                    bm = ball_membership_bits[int(uw)]
                    if candidate_mask is None:
                        candidate_mask = bm
                    else:
                        candidate_mask &= bm
                
                if candidate_mask is not None and candidate_mask != 0:
                    candidate_mask &= ~(1 << c1)
                    candidate_mask &= ~(1 << c2)
                    if candidate_mask != 0:
                        valid_candidates = bits_to_list(candidate_mask)
                        c3 = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                        cc[balls[c3]] += 1
                        # Remove c1, c2 from selected
                        new_selected = [c for c in selected if c != c1 and c != c2]
                        new_selected.append(c3)
                        return new_selected, True
                    else:
                        cc[balls[c1]] += 1
                        cc[balls[c2]] += 1
                else:
                    cc[balls[c1]] += 1
                    cc[balls[c2]] += 1
            else:
                # Both can be removed
                new_selected = [c for c in selected if c != c1 and c != c2]
                return new_selected, True
        
        return selected, False
    
    # Main search loop with multiple restarts
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        
        # Greedy construction
        selected, cc = greedy_cover()
        
        if not selected:
            continue
        
        # Redundancy removal
        selected = remove_redundant(selected, cc)
        
        # 1-swap local search
        selected = one_swap(selected, cc)
        
        # 2-exchange local search
        if len(selected) > 486:
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                num_trials = min(5000, len(selected) * (len(selected) - 1) // 2)
                selected, improved = two_exchange(selected, cc, num_trials)
                if not improved:
                    break
            # After 2-exchange, do another round of swap
            selected = remove_redundant(selected, cc)
            selected = one_swap(selected, cc)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            best_cc = cc.copy()
            
            if best_size <= 486:
                # Do a final polish
                selected = remove_redundant(best_set, best_cc)
                selected = one_swap(selected, best_cc)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
                    best_cc = best_cc.copy()
                
                if best_size <= 403:
                    break
                
                # Try more 2-exchanges on the best solution
                for _ in range(3):
                    if time.time() > deadline - 15.0:
                        break
                    num_trials = min(10000, len(best_set) * (len(best_set) - 1) // 2)
                    best_set, improved = two_exchange(best_set, best_cc, num_trials)
                    if not improved:
                        break
                    best_set = remove_redundant(best_set, best_cc)
                    best_set = one_swap(best_set, best_cc)
                    if len(best_set) < best_size:
                        best_size = len(best_set)
                    if best_size <= 403:
                        break
            else:
                # Continue searching for better solutions
                if best_size <= 500:
                    break
    
    # Convert best_set to output format
    if best_set is None:
        # Fallback: simple greedy
        selected, _ = greedy_cover()
        best_set = selected

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
