import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 50.0 # Leave some margin for 60s limit
    
    def dfs(remainder, min_denom, current_set, current_max):
        nonlocal best_solution, best_max_denom
        
        if time.time() - start_time > time_limit:
            return
        
        if remainder == 0:
            if current_max < best_max_denom:
                best_max_denom = current_max
                best_solution = list(current_set)
            return
        
        if remainder < 0:
            return
            
        # Pruning: if even the smallest possible next term is too small to reach the remainder?
        # No, we need the sum of *all* remaining terms to be at least the remainder.
        # The maximum possible sum of terms with denominators > min_denom and <= best_max_denom
        # is sum(1/d for d in range(min_denom+1, best_max_denom+1)).
        # If this sum < remainder, we can prune.
        # Calculating this sum every time is expensive.
        # Approximation: sum_{d=min_denom+1}^{best_max_denom} 1/d ~ ln(best_max_denom) - ln(min_denom).
        # If ln(best_max_denom) - ln(min_denom) < remainder, prune.
        
        # Also, if 1/min_denom > remainder, we can't use min_denom or any smaller.
        # But we are iterating d from min_denom+1.
        
        # Lower bound for next denominator d:
        # 1/d <= remainder => d >= 1/remainder
        lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
        if lower_bound < min_denom + 1:
            lower_bound = min_denom + 1
            
        # Upper bound for next denominator d:
        # d <= best_max_denom
        # Also, if we pick d, the remainder becomes remainder - 1/d.
        # We must be able to complete the rest with denominators > d.
        
        # Heuristic: try smaller d first? Or larger?
        # To minimize max denom, we are constrained by best_max_denom.
        # We just need to find *any* solution with max < best_max_denom.
        
        for d in range(lower_bound, best_max_denom + 1):
            if time.time() - start_time > time_limit:
                return
            
            next_rem = remainder - Fraction(1, d)
            if next_rem < 0:
                break # Since d is increasing, 1/d is decreasing, so next_rem will increase? 
                # Wait. remainder is fixed. 1/d decreases as d increases.
                # So next_rem = remainder - 1/d increases as d increases.
                # If next_rem < 0, it means 1/d > remainder.
                # As d increases, 1/d decreases, so next_rem will become positive.
                # So we should NOT break. We should continue.
                # However, if d is too small, 1/d > remainder.
                # So we skip these.
                continue
                
            # Pruning check:
            # Can we complete the sum with denominators > d?
            # The maximum possible sum is sum(1/x for x in range(d+1, best_max_denom+1)).
            # If this sum < next_rem, we can't complete.
            # This is a strong prune but expensive to compute.
            # We can approximate.
            
            # Also, if next_rem == 0, we have a solution.
            if next_rem == 0:
                if d < best_max_denom:
                    best_max_denom = d
                    best_solution = list(current_set) + [d]
                return # No need to search further for this branch? 
                       # Actually, we found a solution with max d. 
                       # Since we are iterating d upwards, any other solution in this branch 
                       # would have a larger max denom (since next terms > d). 
                       # So we can return.
            
            # If next_rem > 0, we need more terms.
            # The next term must be > d.
            # If d is already >= best_max_denom, we can't add more terms.
            if d >= best_max_denom:
                continue
                
            # Recursive call
            # We update current_max to max(current_max, d). 
            # But since d > previous denominators, current_max becomes d.
            # However, we are constrained by best_max_denom.
            # So we just pass d as the new min_denom.
            # The "current_max" for pruning is effectively d, because any subsequent term will be > d.
            # So if d >= best_max_denom, we wouldn't have entered the loop (range ends at best_max_denom).
            
            dfs(next_rem, d, current_set + [d], d)

    # Start search
    # First term d1
    # 1/d1 <= 5/31 => d1 >= 31/5 = 6.2 => d1 >= 7
    # d1 < best_max_denom
    for d1 in range(7, best_max_denom + 1):
        if time.time() - start_time > time_limit:
            break
        rem1 = target - Fraction(1, d1)
        if rem1 < 0:
            continue
        if rem1 == 0:
            if d1 < best_max_denom:
                best_max_denom = d1
                best_solution = [d1]
            continue
        dfs(rem1, d1, [d1], d1)
        
    return best_solution
