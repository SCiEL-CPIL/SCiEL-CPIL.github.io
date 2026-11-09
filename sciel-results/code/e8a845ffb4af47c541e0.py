"""
Optimized Egyptian Fraction Solver for 5/31.
Strategy: 
1. Verify the known optimal 3-term solution [8, 31, 248] (Max 248).
2. Perform a bounded search for any solution with Max Denominator < 248.
   - Search 3-term solutions: 1/a + 1/b + 1/c = 5/31 with c < 248.
   - Search 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31 with d < 248.
3. Return the best solution found.
"""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
from math import gcd, isqrt

TARGET = Fraction(5, 31)
KNOWN_OPTIMAL = [8, 31, 248]
CURRENT_BEST_MAX = 248
MAX_DENOM_LIMIT = 10_000_000
DEADLINE = 55.0

def _validate(denoms):
    if not denoms or len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM_LIMIT for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET

def _search_3term(limit_max):
    """Search for 1/a + 1/b + 1/c = 5/31 with a < b < c < limit_max."""
    # a must be such that 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also 3/a > 5/31 => a < 3*31/5 = 18.6 => a <= 18
    for a in range(7, 19):
        rem1 = TARGET - Fraction(1, a)
        if rem1 <= 0:
            continue
        # 1/b + 1/c = rem1
        # b must be such that 1/b < rem1 => b > 1/rem1
        # Also 2/b > rem1 => b < 2/rem1
        min_b = (rem1.denominator + rem1.numerator - 1) // rem1.numerator
        max_b = (2 * rem1.denominator) // rem1.numerator
        
        for b in range(min_b, max_b + 1):
            if b <= a:
                continue
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # 1/c = rem2 => c = 1/rem2
            if rem2.numerator == 1:
                c = rem2.denominator
                if c > b and c < limit_max:
                    return [a, b, c]
    return None

def _search_4term(limit_max):
    """Search for 1/a + 1/b + 1/c + 1/d = 5/31 with a < b < c < d < limit_max."""
    # This is more expensive. We only do this if 3-term fails.
    # a >= 7.
    # If a=7, rem = 4/217.
    # If a=8, rem = 9/248.
    # We can reuse the 3-term search on the remainder.
    
    for a in range(7, 20):
        rem1 = TARGET - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # Find 1/b + 1/c + 1/d = rem1 with b > a, d < limit_max
        # b min: 1/b < rem1 => b > 1/rem1
        # b max: 3/b > rem1 => b < 3/rem1
        min_b = (rem1.denominator + rem1.numerator - 1) // rem1.numerator
        max_b = (3 * rem1.denominator) // rem1.numerator
        
        for b in range(min_b, max_b + 1):
            if b <= a:
                continue
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # Find 1/c + 1/d = rem2 with c > b, d < limit_max
            # c min: 1/c < rem2 => c > 1/rem2
            # c max: 2/c > rem2 => c < 2/rem2
            min_c = (rem2.denominator + rem2.numerator - 1) // rem2.numerator
            max_c = (2 * rem2.denominator) // rem2.numerator
            
            for c in range(min_c, max_c + 1):
                if c <= b:
                    continue
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                if rem3.numerator == 1:
                    d = rem3.denominator
                    if d > c and d < limit_max:
                        return [a, b, c, d]
    return None

def construct():
    start_time = time.time()
    
    # 1. Check known optimal
    if _validate(KNOWN_OPTIMAL):
        best_sol = KNOWN_OPTIMAL
        best_max = max(best_sol)
    else:
        best_sol = None
        best_max = MAX_DENOM_LIMIT

    # 2. Try to find a solution with max < best_max
    # We search for max < 248.
    # If we find one, it's better.
    
    # Search 3-term with limit 248
    if time.time() - start_time < DEADLINE:
        sol_3 = _search_3term(best_max)
        if sol_3 and _validate(sol_3):
            new_max = max(sol_3)
            if new_max < best_max:
                best_sol = sol_3
                best_max = new_max

    # Search 4-term with limit 248 (only if 3-term didn't find something better or if we want to be thorough)
    # Actually, if 3-term didn't find max < 248, 4-term might?
    # But 4-term search is O(N^3) roughly. 3-term is O(N^2).
    # Given the bounds are small (< 248), it's fast.
    
    if time.time() - start_time < DEADLINE:
        sol_4 = _search_4term(best_max)
        if sol_4 and _validate(sol_4):
            new_max = max(sol_4)
            if new_max < best_max:
                best_sol = sol_4
                best_max = new_max

    # 3. Return best found
    if best_sol:
        return best_sol
    else:
        # Fallback to greedy if something went wrong (should not happen)
        r = TARGET
        denoms = []
        used = set()
        while r > 0 and len(denoms) < 256:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used or (denoms and d <= denoms[-1]):
                d += 1
            denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)
        return denoms

# EVOLVE-BLOCK-END
