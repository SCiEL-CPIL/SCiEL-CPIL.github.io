import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def dfs(rem, min_denom, count, path):
        nonlocal best_max, best_result
        if time.time() > deadline:
            return
        
        if rem == 0:
            if len(path) > 0:
                update_best(path)
            return
            
        if rem < 0:
            return
            
        # Lower bound for next denominator
        # 1/x < rem => x > 1/rem
        # Also x >= min_denom
        # If we have 'count' terms left, the max possible sum with distinct denominators starting from min_denom
        # is bounded by harmonic series, but simple bound:
        # If 1/min_denom > rem, no solution? No, we can pick larger denominators.
        # But if 1/min_denom is the largest possible term, and we need 'count' terms...
        
        # Pruning:
        # The smallest possible denominator is min_denom.
        # The largest possible denominator is determined by rem.
        # If 1/min_denom > rem, we must pick a denominator larger than 1/rem.
        # Actually, if 1/min_denom > rem, then min_denom is too small? No, 1/x decreases as x increases.
        # We need 1/x <= rem. So x >= 1/rem.
        
        max_denom = int(1 / rem) + 1
        if max_denom < min_denom:
            return

        # If we have only 1 term left
        if count == 1:
            if rem.denominator == 1:
                x = rem.numerator
                if x >= min_denom and x < best_max:
                    update_best(path + [x])
            return

        # Iterate possible denominators
        # x must be >= min_denom
        # x must be <= 1/rem
        # Also x < best_max (if we want to improve)
        # Optimization: x cannot be too large if we need to sum to rem with 'count' terms.
        # Max sum with 'count' terms starting from x is approx count/x.
        # So count/x >= rem => x <= count/rem.
        
        upper_bound = min(int(count / rem) + 1, best_max)
        
        for x in range(min_denom, upper_bound):
            if time.time() > deadline:
                return
            
            next_rem = rem - Fraction(1, x)
            if next_rem < 0:
                # Since x is increasing, 1/x is decreasing, so next_rem will become more negative?
                # No, rem is fixed. 1/x decreases. next_rem = rem - 1/x increases.
                # So if next_rem < 0, we need smaller x?
                # Wait. rem - 1/x < 0 => 1/x > rem => x < 1/rem.
                # So if x is too small, next_rem is negative.
                # We should start x from ceil(1/rem).
                # But we also have min_denom constraint.
                # So if x < 1/rem, skip.
                continue
                
            # If next_rem == 0, handled by count==1 logic? No, count > 1.
            # If next_rem == 0, we have 0 terms left but count > 1. Invalid.
            if next_rem == 0:
                continue

            dfs(next_rem, x + 1, count - 1, path + [x])

    # Search for 3 terms
    dfs(target, 7, 3, [])
    
    # Search for 4 terms
    if best_max < float('inf'):
        dfs(target, 7, 4, [])
        
    # Search for 5 terms
    if best_max < float('inf'):
        dfs(target, 7, 5, [])

    if best_result is not None:
        return best_result
    
    return [8, 28, 1736]
