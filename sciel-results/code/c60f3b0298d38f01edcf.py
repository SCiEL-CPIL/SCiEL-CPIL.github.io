"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    P_N = 210
    MAX_VAL = 10210
    # Initial best from trivial solution {840}
    best_max = 840
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    # 1. Pair Search: x1 * x2 = 210 * k^2
    # We want min(max(x1, x2)) < best_max
    # Optimal x1 ~ sqrt(210) * k
    # So we check k up to such that sqrt(210)*k < best_max
    max_k = int(math.sqrt(best_max * best_max / 210)) + 10
    # Also k cannot be too large such that min(x1,x2) > 210 is impossible?
    # If 210*k^2 < 210*210, i.e., k^2 < 210, k < 15, no solution.
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        prod = 210 * k * k
        # We need x1, x2 > 210.
        # So prod > 210 * 210 => k^2 > 210 => k >= 15.
        if k < 15:
            continue
        
        # Find divisors of prod close to sqrt(prod)
        # To minimize max, we want x1 as close to sqrt(prod) as possible.
        # x1 must be > 210 and prod/x1 > 210.
        # So 210 < x1 < prod/210.
        # Also x1 != prod/x1.
        
        # Efficient divisor search near sqrt
        root = math.isqrt(prod)
        # Check downwards from root
        found_pair = None
        # We want the pair with min max.
        # Max is determined by the larger element.
        # If x1 <= sqrt(prod), x2 >= sqrt(prod). Max is x2 = prod/x1.
        # To minimize max, we need to minimize x2, which means maximizing x1 (subject to x1 <= sqrt(prod)).
        # So we look for the largest divisor x1 <= sqrt(prod) such that x1 > 210 and prod/x1 > 210.
        
        # Lower bound for x1: 211
        # Upper bound for x1: min(root, prod//211)
        upper_x1 = min(root, prod // 211)
        
        # Iterate x1 downwards from upper_x1 to 211
        # This might be slow if range is large, but range is roughly sqrt(prod) - 210.
        # For k=15, sqrt(47250) ~ 217. Range is tiny.
        # For k=50, sqrt(210*2500) ~ 725. Range ~ 500.
        # This is fast.
        
        for x1 in range(upper_x1, 210, -1):
            if prod % x1 == 0:
                x2 = prod // x1
                if x2 > 210 and x1 != x2:
                    # Valid pair
                    curr_max = max(x1, x2)
                    if curr_max < best_max:
                        best_max = curr_max
                        # Update max_k based on new best_max
                        max_k = int(math.sqrt(best_max * best_max / 210)) + 10
                    break # Found best for this k, move to next k

    # 2. Triple Search: x1 * x2 * x3 = 210 * m^2
    # We want min(max(x1, x2, x3)) < best_max
    # Optimal x_i ~ (210)^(1/3) * m^(2/3)
    # Check m such that optimal < best_max
    max_m = int((best_max / (210 ** (1.0/3.0))) ** 1.5) + 10
    
    for m in range(1, max_m + 1):
        if time.time() > deadline:
            break
        prod = 210 * m * m
        # Need x1, x2, x3 > 210.
        # Min product > 210^3.
        # 210 m^2 > 210^3 => m^2 > 210^2 => m > 210.
        if m <= 210:
            continue
            
        # We want x1, x2, x3 distinct, > 210, product = prod.
        # Minimize max.
        # Iterate x1 from 211 to cbrt(prod)
        # Iterate x2 from x1+1 to sqrt(prod/x1)
        # Calculate x3
        
        cbrt_prod = int(round(prod ** (1.0/3.0)))
        # Ensure cbrt_prod is not too large
        if cbrt_prod > best_max:
            cbrt_prod = best_max - 1
            
        for x1 in range(211, cbrt_prod + 1):
            if prod % x1 != 0:
                continue
            rem1 = prod // x1
            # x2 * x3 = rem1
            # x2 > x1, x3 > x2
            # x2 < sqrt(rem1)
            sqrt_rem1 = math.isqrt(rem1)
            if sqrt_rem1 >= best_max:
                sqrt_rem1 = best_max - 1
                
            for x2 in range(x1 + 1, sqrt_rem1 + 1):
                if rem1 % x2 != 0:
                    continue
                x3 = rem1 // x2
                if x3 <= x2:
                    continue
                if x3 > 210 and x3 < best_max:
                    curr_max = x3
                    if curr_max < best_max:
                        best_max = curr_max
                        # Update max_m
                        max_m = int((best_max / (210 ** (1.0/3.0))) ** 1.5) + 10
                        break
            if time.time() > deadline:
                break

    # Return the best solution found
    # We need to reconstruct the set.
    # If best_max is still 840, return [840].
    # Otherwise, we need to store the actual numbers during search.
    
    # Let's refactor to store the solution.
    # Re-run logic or store during first pass?
    # Storing is better.
    
    best_sol = [840]
    
    # Pair Search again, storing solution
    max_k = int(math.sqrt(best_max * best_max / 210)) + 10
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        prod = 210 * k * k
        if k < 15:
            continue
        root = math.isqrt(prod)
        upper_x1 = min(root, prod // 211)
        for x1 in range(upper_x1, 210, -1):
            if prod % x1 == 0:
                x2 = prod // x1
                if x2 > 210 and x1 != x2:
                    curr_max = max(x1, x2)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_sol = [x1, x2]
                        max_k = int(math.sqrt(best_max * best_max / 210)) + 10
                    break

    # Triple Search again, storing solution
    max_m = int((best_max / (210 ** (1.0/3.0))) ** 1.5) + 10
    for m in range(1, max_m + 1):
        if time.time() > deadline:
            break
        prod = 210 * m * m
        if m <= 210:
            continue
        cbrt_prod = int(round(prod ** (1.0/3.0)))
        if cbrt_prod > best_max:
            cbrt_prod = best_max - 1
        for x1 in range(211, cbrt_prod + 1):
            if prod % x1 != 0:
                continue
            rem1 = prod // x1
            sqrt_rem1 = math.isqrt(rem1)
            if sqrt_rem1 >= best_max:
                sqrt_rem1 = best_max - 1
            for x2 in range(x1 + 1, sqrt_rem1 + 1):
                if rem1 % x2 != 0:
                    continue
                x3 = rem1 // x2
                if x3 <= x2:
                    continue
                if x3 > 210 and x3 < best_max:
                    curr_max = x3
                    if curr_max < best_max:
                        best_max = curr_max
                        best_sol = [x1, x2, x3]
                        max_m = int((best_max / (210 ** (1.0/3.0))) ** 1.5) + 10
                        break
            if time.time() > deadline:
                break

    return best_sol
# EVOLVE-BLOCK-END
