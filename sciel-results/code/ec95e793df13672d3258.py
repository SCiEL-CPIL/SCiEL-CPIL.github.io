import time
from fractions import Fraction
from math import gcd

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _precompute_recip_sums(limit):
    sums = [Fraction(0, 1)] * (limit + 1)
    for i in range(1, limit + 1):
        sums[i] = sums[i-1] + Fraction(1, i)
    return sums

# We search for max < 248. So denominators up to 247.
# Precompute up to 250.
RECIP_SUMS = _precompute_recip_sums(250)

def _max_sum(min_d, max_d):
    if min_d > max_d:
        return Fraction(0, 1)
    if min_d <= 0:
        min_d = 1
    if max_d > len(RECIP_SUMS) - 1:
        return sum(Fraction(1, i) for i in range(min_d, max_d + 1))
    return RECIP_SUMS[max_d] - RECIP_SUMS[min_d - 1]

def _find_optimal_solution(limit):
    """
    Find the solution with the minimum possible max denominator < limit.
    """
    best_max = [limit]
    best_sol = [None]
    
    def dfs(remaining, min_d, current_max, path):
        # Pruning
        if current_max >= best_max[0]:
            return
            
        if remaining == 0:
            if current_max < best_max[0]:
                best_max[0] = current_max
                best_sol[0] = path[:]
            return
            
        # If remaining is too small to be formed by any term <= limit-1
        if remaining < Fraction(1, limit - 1):
            return
            
        # Max possible sum with denominators [min_d, limit-1]
        max_possible = _max_sum(min_d, limit - 1)
        if remaining > max_possible:
            return
            
        # Lower bound for d
        inv_rem = 1 / remaining
        lower_d = max(min_d, inv_rem.numerator // inv_rem.denominator + (1 if inv_rem.numerator % inv_rem.denominator != 0 else 0))
        
        upper_d = limit - 1
        
        for d in range(lower_d, upper_d + 1):
            if d >= best_max[0]:
                break
                
            term = Fraction(1, d)
            if term > remaining:
                continue
                
            new_rem = remaining - term
            
            if new_rem == 0:
                new_max = max(current_max, d)
                if new_max < best_max[0]:
                    best_max[0] = new_max
                    best_sol[0] = path + [d]
                # Do not return, continue searching for potentially smaller max
                # But if d is increasing, and we found a solution, 
                # any subsequent d' > d will have max >= d > current_max?
                # Not necessarily. But we can break if d >= best_max[0]?
                # We already have `if d >= best_max[0]: break`.
                # So if we found a solution, best_max[0] is updated.
                # The loop condition `if d >= best_max[0]: break` will handle it.
                
            else:
                # new_rem > 0
                if d + 1 > limit - 1:
                    continue
                
                # Check if new_rem can be formed by denominators [d+1, limit-1]
                max_rest_sum = _max_sum(d + 1, limit - 1)
                if new_rem > max_rest_sum:
                    continue
                    
                # Smallest term we can add next is 1/(limit-1)
                # If new_rem < 1/(limit-1), we can't form it
                if new_rem < Fraction(1, limit - 1):
                    continue
                
                new_max = max(current_max, d)
                if new_max >= best_max[0]:
                    continue
                    
                path.append(d)
                dfs(new_rem, d + 1, new_max, path)
                path.pop()
                
    dfs(TARGET, 1, 0, [])
    
    if best_sol[0]:
        return best_sol[0]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Known solution
    best_result = [8, 31, 248]
    best_max = 248
    
    # Search for solution with max < 248
    if time.time() < deadline - 5:
        sol = _find_optimal_solution(248)
        if sol:
            max_d = max(sol)
            if max_d < best_max:
                best_result = sol
                best_max = max_d
    
    return best_result
