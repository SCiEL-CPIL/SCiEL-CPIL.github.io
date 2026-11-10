import math
import time
import bisect

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute all divisors of N that are > 12
    all_divisors = set()
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            all_divisors.add(i)
            all_divisors.add(N // i)
    
    # Filter to divisors > 12
    valid_divisors = sorted([d for d in all_divisors if d > 12])
    
    best_factors = [N]
    best_max_val = N
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    def time_exceeded():
        return time.time() > deadline
    
    def can_complete(target_rem, remaining, last_factor):
        """Check if target_rem can be factored into 'remaining' distinct integers > last_factor and > 12."""
        if remaining == 1:
            return target_rem > last_factor and target_rem > 12
        if remaining == 2:
            # Need two distinct integers a, b > last_factor with a*b = target_rem, a < b
            a_max = int(math.isqrt(target_rem))
            for a in range(max(last_factor + 1, 13), a_max + 1):
                if target_rem % a == 0:
                    b = target_rem // a
                    if b > a and b > last_factor and b > 12:
                        return True
            return False
        # For remaining >= 3: minimum possible product of 'remaining' distinct ints > last_factor
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
        if target_rem < min_prod:
            return False
        # Also: target_rem must be divisible by at least 'remaining' distinct factors > last_factor
        # Quick check: count how many valid divisors of target_rem are > last_factor
        count = 0
        for d in valid_divisors:
            if d > last_factor and target_rem % d == 0:
                count += 1
                if count >= remaining:
                    break
        if count < remaining:
            return False
        return True
    
    def dfs(current_prod, last_factor, depth, target_k, path):
        nonlocal best_max_val, best_factors
        
        if time_exceeded():
            return
        
        if depth == target_k:
            if current_prod == N:
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        target_rem = N // current_prod
        
        # Feasibility check
        if not can_complete(target_rem, remaining, last_factor):
            return
        
        # If remaining == 1, the next factor must be target_rem
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12:
                path.append(x)
                dfs(current_prod * x, x, depth + 1, target_k, path)
                path.pop()
            return
        
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound: x * (x+1) * ... * (x+remaining-1) <= target_rem
        # So x <= target_rem^(1/remaining)
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        
        # Additional pruning: the largest factor in any solution can't exceed best_max_val
        # If we're trying to beat best_max_val, then every factor must be < best_max_val
        if best_max_val < N:
            x_upper = min(x_upper, best_max_val - 1)
        
        if x_upper < lower_bound:
            return
        
        # Find divisors of target_rem in [lower_bound, x_upper] that are also divisors of N
        left_idx = bisect.bisect_left(valid_divisors, lower_bound)
        right_idx = bisect.bisect_right(valid_divisors, x_upper)
        
        candidates = []
        for idx in range(left_idx, right_idx):
            x = valid_divisors[idx]
            if x > last_factor and x > 12 and target_rem % x == 0:
                candidates.append(x)
        
        # Sort by closeness to geometric mean for balanced solutions
        if candidates:
            ideal = target_rem ** (1.0 / remaining)
            candidates.sort(key=lambda x: abs(math.log(x) - math.log(ideal)))
        
        for x in candidates:
            if time_exceeded():
                return
            path.append(x)
            dfs(current_prod * x, x, depth + 1, target_k, path)
            path.pop()
            # Prune: if we've found a solution with max < x_upper, no point continuing
            if best_max_val <= x_upper:
                break
    
    # Try different numbers of factors. More factors generally means smaller max.
    # Start with more factors and work down.
    for k in range(7, 1, -1):
        dfs(1, 12, 0, k, [])
        if best_max_val <= 100:
            break
    
    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
