from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # Target sum
    target = Fraction(1, 1)
    
    # We want to find N terms. Try N=8 first.
    # If not found, try N=9, etc.
    # But N=8 is the minimum possible.
    
    def search(remaining_sum, min_den, terms_left, current_denoms):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_sum == 0:
                return current_denoms
            return None
        
        if terms_left == 1:
            if remaining_sum > 0:
                d = 1 / remaining_sum
                if d == int(d):
                    d_int = int(d)
                    if d_int >= min_den and d_int <= 10000000:
                        return current_denoms + [d_int]
            return None
        
        # Pruning
        # Max possible sum with terms_left terms, starting from min_den
        # Approximation: terms_left / min_den
        if remaining_sum > Fraction(terms_left, min_den):
            return None
        
        # Min possible sum with terms_left terms, ending at 10^7
        # Approximation: terms_left / 10000000
        if remaining_sum < Fraction(terms_left, 10000000):
            return None
        
        # Determine max denominator to try
        # 1/d <= remaining_sum / terms_left  => d >= terms_left / remaining_sum
        # Also d > previous denom
        # Also 1/d < remaining_sum (since other terms are positive)
        # So d > 1 / remaining_sum
        
        max_d = 10000000
        # We can restrict d further.
        # If we pick d, remaining_sum - 1/d must be achievable by terms_left-1 terms > d.
        # Max sum of next terms: (terms_left-1) / (d+1)
        # So remaining_sum - 1/d <= (terms_left-1) / (d+1)
        # This inequality helps bound d from below or above?
        # Actually, we iterate d from min_den upwards.
        
        # Upper bound for d in this step:
        # 1/d < remaining_sum => d > 1/remaining_sum
        # Also, we need to leave enough sum for the rest.
        # Min sum of rest: (terms_left-1) / 10000000
        # So 1/d >= remaining_sum - (terms_left-1)/10000000
        # d <= 1 / (remaining_sum - (terms_left-1)/10000000)
        
        lower_bound = min_den
        upper_bound = 10000000
        
        # Refine upper bound
        # remaining_sum - 1/d >= (terms_left-1) / max_possible_next_denom
        # This is hard to use directly for upper bound of d.
        
        # Let's just iterate d from min_den.
        # To speed up, we can jump.
        # But for N=8, the tree is small.
        
        for d in range(min_den, upper_bound + 1):
            if time.time() > deadline:
                return None
                
            # If d is too large, 1/d is too small to help?
            # Actually, if 1/d > remaining_sum, we break.
            if Fraction(1, d) >= remaining_sum:
                break
            
            # Check if remaining sum is feasible with d+1..
            new_rem = remaining_sum - Fraction(1, d)
            
            # Feasibility check for next step
            if new_rem <= 0:
                continue
                
            # Max sum of next terms: (terms_left-1) / (d+1)
            if new_rem > Fraction(terms_left - 1, d + 1):
                # If new_rem is larger than max possible sum of remaining terms,
                # we need a smaller d (larger 1/d).
                # But we are iterating d upwards, so 1/d is decreasing.
                # If current d is too small (1/d too large), we might have overshot?
                # No, 1/d < remaining_sum.
                # If new_rem > max_possible, it means we subtracted too little?
                # No, if we subtract 1/d, and the rest can't sum to new_rem,
                # it means new_rem is too big for the remaining terms.
                # This happens if d is too small (1/d too big? No, 1/d small means d big).
                # Wait.
                # If d is small, 1/d is large. new_rem is small.
                # If d is large, 1/d is small. new_rem is large.
                # So if new_rem > max_possible, d is too large.
                # We can break the loop?
                # As d increases, 1/d decreases, new_rem increases.
                # So if new_rem is already too big, it will stay too big.
                # So we can break.
                break
            
            # Min sum of next terms: (terms_left-1) / 10000000
            if new_rem < Fraction(terms_left - 1, 10000000):
                # new_rem is too small.
                # This happens if d is too small (1/d too large).
                # As d increases, new_rem increases.
                # So we might reach a feasible range.
                # Continue.
                pass
            
            res = search(new_rem, d + 1, terms_left - 1, current_denoms + [d])
            if res:
                return res
        
        return None

    # Try N=8
    result = search(Fraction(1, 1), 5, 8, [])
    if result:
        return result
        
    # Fallback to N=9 if N=8 fails (unlikely)
    result = search(Fraction(1, 1), 5, 9, [])
    if result:
        return result
        
    # Fallback to original DP if search fails
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no solution found")
