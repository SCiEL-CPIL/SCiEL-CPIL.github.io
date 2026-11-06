"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

_MAX_DEN = 10000000


def _max_fill_float(last, count):
    """Approximate max reciprocal sum with `count` distinct denominators > last."""
    if count <= 0:
        return 0.0
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (last + j)
    return s


def _min_fill_float(last, count):
    """Approximate min reciprocal sum with `count` distinct denominators > last (largest possible)."""
    if count <= 0:
        return 0.0
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (_MAX_DEN - count + j)
    return s


def _greedy_egyptian(remaining_num, remaining_den, last, deadline):
    """Greedy search using integer arithmetic for speed.
    remaining = remaining_num / remaining_den
    Returns list of denominators or None."""
    if remaining_num == 0:
        return []
    
    # Check if remaining is exactly 1/d for some d
    if remaining_num == 1:
        d = remaining_den
        if d > last and d >= 5 and d <= _MAX_DEN:
            return [d]
        return None
    
    # Find smallest d > last such that 1/d <= remaining
    # 1/d <= remaining_num/remaining_den  =>  d >= remaining_den/remaining_num
    d_from_frac = remaining_den // remaining_num
    if d_from_frac * remaining_num < remaining_den:
        d_from_frac += 1
    d_low = max(last + 1, 5, d_from_frac)
    
    d = d_low
    guard = 0
    while d <= _MAX_DEN and guard < 200000:
        guard += 1
        if time.time() > deadline:
            return None
        
        # Check 1/d <= remaining: d * remaining_num >= remaining_den
        if d * remaining_num < remaining_den:
            d += 1
            continue
        
        # new_remaining = remaining - 1/d = (remaining_num*d - remaining_den) / (remaining_den*d)
        nr_num = remaining_num * d - remaining_den
        nr_den = remaining_den * d
        
        if nr_num < 0:
            d += 1
            continue
        
        # Simplify
        g = math.gcd(nr_num, nr_den)
        nr_num //= g
        nr_den //= g
        
        # Feasibility check: can we represent nr_num/nr_den with (remaining_num-1) terms > d?
        # Max possible: sum of 1/(d+1) + 1/(d+2) + ... + 1/(d+(remaining_num-1))
        # Min possible: sum of 1/(_MAX_DEN) + ... (very small, usually not binding)
        
        max_possible = 0.0
        for j in range(1, remaining_num):
            max_possible += 1.0 / (d + j)
        
        if nr_num / nr_den > max_possible + 1e-15:
            # Even taking the smallest denominators, we can't reach this high
            # As d increases, 1/d decreases, so nr increases - this gets worse
            return None
        
        min_possible = 0.0
        for j in range(1, remaining_num):
            min_possible += 1.0 / (_MAX_DEN - remaining_num + 1 + j)
        
        if nr_num / nr_den < min_possible - 1e-15:
            # Too small to be represented; increase d to make nr larger
            d += 1
            continue
        
        # Recursive call
        res = _greedy_egyptian(nr_num, nr_den, d, deadline)
        if res is not None:
            return [d] + res
        
        d += 1
    
    return None


def _search(k, deadline):
    """Try to find exactly k distinct denominators >= 5 summing to 1."""
    # We want sum of k reciprocals = 1, so remaining = 1/1 initially
    return _greedy_egyptian(1, 1, 4, deadline)


def _known_solution():
    """Return a known valid solution with small count."""
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is too many. Let's use a smaller known decomposition.
    # Actually, let's just return a fallback that's guaranteed to work.
    pass


def _fallback():
    """Guaranteed-valid baseline solution using subset DP over divisors of a common denominator."""
    # Use a larger target for more flexibility
    for target in [2520, 5040, 10080, 25200, 50400, 100800, 252000, 504000]:
        reached = {0: []}
        max_d = min(target + 1, 100001)
        for d in range(5, max_d):
            if target % d != 0:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and (total + w) not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
    
    raise RuntimeError("no solution found")


def construct():
    deadline = time.time() + 55.0
    
    # Try greedy for increasing k starting from 5
    # Start from smaller k to find minimal solutions
    for k in range(5, 30):
        if time.time() > deadline:
            break
        sol = _search(k, deadline)
        if sol is not None:
            # Verify the solution
            total = Fraction(0, 1)
            for d in sol:
                total += Fraction(1, d)
            if total == 1 and len(sol) == len(set(sol)) and all(5 <= d <= _MAX_DEN for d in sol):
                return sol
    
    return _fallback()
# EVOLVE-BLOCK-END
