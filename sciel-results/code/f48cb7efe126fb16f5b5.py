"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from math import factorial

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill
    
    N = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1
    
    # Prime exponents of N
    target_exp = {2: 10, 3: 5, 5: 2, 7: 1}
    
    # Generate all divisors of N
    divs = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divs.append(i)
            if i != N // i:
                divs.append(N // i)
        i += 1
    divs = sorted(set(divs))
    
    # Filter divisors > 12
    valid_divs = [d for d in divs if d > 12]
    
    best_max = N
    best_factors = [N]
    
    # DFS with pruning
    # State: remaining exponent vector, current factor list, min next factor, current max
    sys.setrecursionlimit(10000)
    
    # Precompute: for each divisor, its prime exponents
    def factorize_exp(d):
        exp = {2: 0, 3: 0, 5: 0, 7: 0}
        temp = d
        for p in [2, 3, 5, 7]:
            while temp % p == 0:
                exp[p] += 1
                temp //= p
        return exp
    
    div_exp = {d: factorize_exp(d) for d in valid_divs}
    
    # Cache for feasibility checks
    feasibility_cache = {}
    
    def feasible(rem_exp, min_factor, max_factors_left):
        """Check if rem_exp can be formed by at most max_factors_left distinct integers >= min_factor, each > 12."""
        # Key for caching
        key = (tuple(rem_exp[p] for p in [2,3,5,7]), min_factor, max_factors_left)
        if key in feasibility_cache:
            return feasibility_cache[key]
        
        # Convert rem_exp to integer
        rem_val = 1
        for p in [2, 3, 5, 7]:
            rem_val *= (p ** rem_exp[p])
        
        if rem_val == 1:
            result = True
        elif max_factors_left <= 0:
            result = False
        else:
            # If we use exactly one more factor, it must be rem_val itself
            # and it must be >= min_factor and > 12
            if max_factors_left == 1:
                result = (rem_val >= min_factor and rem_val > 12)
            else:
                # Try to find at least one valid first factor
                found = False
                # Iterate through valid divisors that divide rem_val
                # and are >= min_factor
                for d in valid_divs:
                    if d < min_factor:
                        break
                    if d > rem_val:
                        break
                    if rem_val % d != 0:
                        continue
                    other = rem_val // d
                    # The remaining must be formable by at most max_factors_left - 1 factors
                    # all >= d+1 (since distinct)
                    # But we need to check if other's prime exponents are <= rem_exp
                    oe = {}
                    temp = other
                    for p in [2,3,5,7]:
                        cnt = 0
                        while temp % p == 0:
                            cnt += 1
                            temp //= p
                        oe[p] = cnt
                    # Check oe <= rem_exp
                    valid = True
                    for p in [2,3,5,7]:
                        if oe[p] > rem_exp[p]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # New remaining exp
                    new_rem = {p: rem_exp[p] - oe[p] for p in [2,3,5,7]}
                    # Recurse
                    if feasible(new_rem, d + 1, max_factors_left - 1):
                        found = True
                        break
                result = found
        
        feasibility_cache[key] = result
        return result
    
    def dfs(rem_exp, current_factors, min_next, current_max):
        nonlocal best_max, best_factors, t0
        
        if time.time() - t0 > DEADLINE:
            return
        
        rem_val = 1
        for p in [2, 3, 5, 7]:
            rem_val *= (p ** rem_exp[p])
        
        if rem_val == 1:
            # We have a complete factorization
            if len(current_factors) > 0:
                cur_max = current_max
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = list(current_factors)
            return
        
        # Number of factors already used
        used = len(current_factors)
        max_left = 12 - used
        
        if max_left <= 0:
            return
        
        # Pruning: if rem_val < min_next, we can't form it with factors >= min_next
        if rem_val < min_next:
            return
        
        # Check if feasible with remaining budget
        if not feasible(rem_exp, min_next, max_left):
            return
        
        # Try divisors d of rem_val, d >= min_next
        # We iterate valid_divs in order
        for d in valid_divs:
            if d < min_next:
                continue
            if d > rem_val:
                break
            if rem_val % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                # If we've already found something better, and this factor alone
                # makes it no better, skip. But we still need to check if remaining
                # can be formed... actually if new_max >= best_max, any completion
                # will have max >= new_max >= best_max, so no improvement.
                break
            
            # Check d's exponents
            de = div_exp[d]
            valid = True
            for p in [2, 3, 5, 7]:
                if de[p] > rem_exp[p]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Subtract
            new_rem = {p: rem_exp[p] - de[p] for p in [2, 3, 5, 7]}
            
            current_factors.append(d)
            dfs(new_rem, current_factors, d + 1, new_max)
            current_factors.pop()
        
        return
    
    # Start DFS
    dfs(dict(target_exp), [], 13, 0)
    
    # Return the best solution found
    if best_factors:
        return best_factors
    else:
        return [N]

# EVOLVE-BLOCK-END
