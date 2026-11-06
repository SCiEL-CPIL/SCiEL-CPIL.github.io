"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from itertools import combinations

def _greedy_complete(r, used, max_d=10000000):
    """Complete r = sum 1/d_i using greedy, avoiding used denominators.
    Returns list of denominators or None if stuck."""
    result = []
    r = Fraction(r)
    while r > 0:
        if r.denominator == 1:
            # r is an integer, need 1/r terms of 1/1? No, r is a fraction.
            # Actually if r is integer, 1/d where d=1/r? No.
            # r = a/b. We need 1/d such that 1/d <= r.
            pass
        # Find smallest d not in used such that 1/d <= r
        # 1/d <= a/b => d >= b/a
        d_min = (r.denominator + r.numerator - 1) // r.numerator
        if d_min < 1:
            d_min = 1
        found = False
        # Try a few candidates around d_min
        # Since we want to minimize max denominator, greedy picks the smallest possible d.
        # But we might skip a d to get a smaller max later? Greedy is usually good.
        # Let's just try the smallest available.
        for d in range(d_min, d_min + 1000):
            if d > max_d:
                break
            if d in used:
                continue
            if Fraction(1, d) > r:
                continue
            result.append(d)
            used.add(d)
            r -= Fraction(1, d)
            found = True
            break
        if not found:
            return None
    return result

def _check_solution(denoms, target):
    s = sum(Fraction(1, d) for d in denoms)
    return s == target

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    target = Fraction(5, 31)
    max_d_limit = 10000000
    
    best_solution = None
    best_max = float('inf')
    
    # Initial greedy solution
    r = target
    used = set()
    init_denoms = _greedy_complete(r, used, max_d_limit)
    if init_denoms:
        best_solution = init_denoms
        best_max = max(init_denoms)
    
    # Search for better solutions
    # Strategy 1: Fix first 1 or 2 denominators, complete the rest with greedy.
    # Since we want to minimize the max, we should try small first denominators.
    
    # Try 2-term prefixes
    # 1/d1 + 1/d2 + ... = 5/31
    # 1/d1 < 5/31 => d1 > 31/5 = 6.2 => d1 >= 7
    # 1/d2 < 5/31 - 1/d1
    
    d1_max_search = 50  # Search up to 50 for d1
    d2_max_search = 500 # Search up to 500 for d2
    
    for d1 in range(7, d1_max_search + 1):
        if time.time() > deadline:
            break
        if 1.0/d1 >= 5.0/31.0:
            continue
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        # Complete r1 with greedy
        used = {d1}
        denoms = [d1] + _greedy_complete(r1, used, max_d_limit)
        if denoms:
            if _check_solution(denoms, target):
                m = max(denoms)
                if m < best_max:
                    best_max = m
                    best_solution = denoms
        
        # Try fixing d2 as well
        # 1/d2 < r1 => d2 > 1/r1
        d2_min = int(1 / float(r1)) + 1
        if d2_min < d1 + 1:
            d2_min = d1 + 1
        
        for d2 in range(d2_min, d2_max_search + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            used = {d1, d2}
            denoms = [d1, d2] + _greedy_complete(r2, used, max_d_limit)
            if denoms:
                if _check_solution(denoms, target):
                    m = max(denoms)
                    if m < best_max:
                        best_max = m
                        best_solution = denoms
                        
    # Strategy 2: Local search / Perturbation
    # If we have a solution, can we reduce the max denominator?
    # This is hard. Let's stick to the prefix search.
    
    # Strategy 3: Check if best_solution is valid and within bounds
    if best_solution is None:
        # Fallback to simple greedy if something went wrong
        r = target
        used = set()
        best_solution = _greedy_complete(r, used, max_d_limit)
        if best_solution:
            best_max = max(best_solution)
    
    return best_solution

# EVOLVE-BLOCK-END
