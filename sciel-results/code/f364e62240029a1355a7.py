import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

# Precomputed small solutions: (denominators, count)
# We want the minimum count of distinct denominators in [5, 10000000]
# whose reciprocal sum is exactly 1.

# For 4 terms: 1/5 + 1/7 + 1/9 + 1/10 = (630+450+350+315)/3150 = 1745/3150 != 1
# Let's find actual minimal solutions.

# Known: 1 = 1/5 + 1/7 + 1/9 + 1/10 + ... no, let me compute properly.

# Let's search for small solutions programmatically at import time,
# but with a strict time budget.

def _find_solution_3_terms():
    """Find 3 distinct denominators in [5, 10000000] with reciprocal sum = 1."""
    # 1/a + 1/b + 1/c = 1 with 5 <= a < b < c
    # c = ab / (ab - a - b) must be integer
    # So ab - a - b must divide ab
    # (a-1)(b-1) = ab - a - b + 1
    # Let k = ab - a - b, need k | ab
    # k = ab - a - b
    # ab/k = 1 + (a+b)/k
    # For this to be integer, k must divide ab.
    # k = ab - a - b. Note that a+b+1 = k + 1.
    # ab/k = ab/(ab-a-b). Let g = gcd(ab, ab-a-b) = gcd(ab, a+b).
    # So we need ab/(ab-a-b) integer, i.e., (ab-a-b) | ab.
    # Since (ab-a-b) = ab - (a+b), we need (a+b) | (ab-a-b) + (a+b) = ab.
    # Wait: (ab-a-b) | ab iff (ab-a-b) | (ab + a + b - (ab-a-b)) = 2(a+b)
    # Hmm, let me just do it directly: need (a-1)(b-1) - 1 divides... 
    # Actually: c = ab/(ab-a-b). Need ab-a-b > 0 (i.e., a,b >= 3) and (ab-a-b) | ab.
    # Let d = ab - a - b. Need d | ab. Note d = (a-1)(b-1) - 1.
    # gcd(d, ab) = gcd(ab-a-b, ab) = gcd(a+b, ab) (since ab - (ab-a-b) = a+b)
    # So need d | ab, which means d | gcd-related... just iterate.
    
    # For a in [5, ...]: b must be such that ab - a - b > 0 and (ab-a-b) | ab
    # ab - a - b > 0 => b(a-1) > a => b > a/(a-1) which is always true for b >= 5, a >= 5
    # But also c > b, so ab/(ab-a-b) > b => a/(ab-a-b) > 1 => a > ab-a-b => 2a+b > ab => b(a-2) < 2a => b < 2a/(a-2)
    # For a=5: b < 10/3 = 3.33... but b > a = 5, contradiction!
    # For a=6: b < 12/4 = 3, contradiction.
    # For a=7: b < 14/5 = 2.8, contradiction.
    # So NO 3-term solutions with all denominators >= 5!
    return None

def _find_solution_4_terms():
    """Find 4 distinct denominators in [5, 10000000] with reciprocal sum = 1."""
    # 1/a + 1/b + 1/c + 1/d = 1, 5 <= a < b < c < d
    # Fix a, b, c, solve for d: d = 1 / (1 - 1/a - 1/b - 1/c)
    # Need 1 - 1/a - 1/b - 1/c > 0 and d integer > c
    deadline = time.time() + 5.0
    
    for a in range(5, 20):
        if time.time() > deadline:
            break
        for b in range(a+1, 50):
            s2 = 1.0 - 1.0/a - 1.0/b
            if s2 <= 0:
                break
            for c in range(b+1, 200):
                s3 = s2 - 1.0/c
                if s3 <= 0:
                    break
                if s3 > 0.5:  # would give d < 2, impossible
                    continue
                # d = 1/s3 must be integer > c
                # 1/s3 = 1 / (1 - 1/a - 1/b - 1/c)
                # Let's use exact fractions
                from fractions import Fraction
                f = Fraction(1,1) - Fraction(1,a) - Fraction(1,b) - Fraction(1,c)
                if f.numerator == 1:
                    d = f.denominator
                    if d > c and d <= 10000000:
                        return [a, b, c, d]
                elif f.numerator > 1:
                    if f.denominator % f.numerator == 0:
                        d = f.denominator // f.numerator
                        if d > c and d <= 10000000:
                            return [a, b, c, d]
    return None

def _find_solution_5_terms():
    """Find 5 distinct denominators in [5, 10000000] with reciprocal sum = 1."""
    from fractions import Fraction
    deadline = time.time() + 8.0
    
    for a in range(5, 12):
        if time.time() > deadline:
            break
        for b in range(a+1, 25):
            s2 = Fraction(1,1) - Fraction(1,a) - Fraction(1,b)
            if s2 <= 0:
                break
            for c in range(b+1, 50):
                s3 = s2 - Fraction(1,c)
                if s3 <= 0:
                    break
                if s3 >= Fraction(1,2):  # remaining 2 terms can't reach 1/2 if both > c
                    # Actually 1/d + 1/e <= 2/(c+1) roughly, but let's just continue
                    pass
                for d in range(c+1, 100):
                    if time.time() > deadline:
                        break
                    s4 = s3 - Fraction(1,d)
                    if s4 <= 0:
                        break
                    if s4 >= Fraction(1, c+1):  # 1/e <= 1/(d+1) < 1/c
                        continue
                    # e = 1/s4 must be integer > d
                    if s4.numerator == 1:
                        e = s4.denominator
                        if e > d and e <= 10000000:
                            return [a, b, c, d, e]
                    elif s4.denominator % s4.numerator == 0:
                        e = s4.denominator // s4.numerator
                        if e > d and e <= 10000000:
                            return [a, b, c, d, e]
    return None

def _precompute_best():
    """Find the minimum-count solution at module load."""
    start = time.time()
    deadline = start + 10.0  # strict internal deadline
    
    # Try 3 terms first (should fail quickly)
    sol3 = _find_solution_3_terms()
    if sol3 is not None:
        return sol3
    
    if time.time() > deadline:
        return None
    
    # Try 4 terms
    sol4 = _find_solution_4_terms()
    if sol4 is not None:
        return sol4
    
    if time.time() > deadline:
        return None
    
    # Try 5 terms
    sol5 = _find_solution_5_terms()
    if sol5 is not None:
        return sol5
    
    return None

# Attempt precomputation with a hard cap
_best_precomputed = None
_t0 = time.time()
if time.time() < _t0 + 25:  # leave margin for evaluator's 90s limit
    _best_precomputed = _precompute_best()

# Fallback: known 4-term solution if search failed
# 1/5 + 1/6 + 1/7 + 1/8 + 1/140? No...
# Let's verify: 1/5+1/6+1/7+1/8 = (168+140+120+105)/840 = 533/840. 1 - 533/840 = 307/840. Not unit.
# Known: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + ... 
# Actually a known 4-term: 1/5 + 1/7 + 1/9 + 1/10 = (630+450+350+315)/3150 = 1745/3150. No.
# Let me just use a verified one from the search or a hardcoded fallback.

def _hardcoded_fallback():
    # Verify: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/15 + 1/20 + 1/25 + 1/28 + 1/30 + 1/35 + 1/40 + 1/42 + 1/45 + 1/56 + 1/60 + 1/70 + 1/84 + 1/90
    # That's too many. Let's use the precomputed or a simple valid one.
    # Simple valid: 1/5 + 1/5+1/5+1/5 + 1/5+1/5+1/5+1/5 = no, need distinct.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + ... 
    # Just use a definitely valid large solution as last resort
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Sum of 1/5..1/30 ≈ ln(30)-ln(4) + small = ln(7.5) ≈ 2.01, too much.
    # Better: just use the precomputed result or raise.
    return None

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1; minimize their count."""
    # Use precomputed best if available
    if _best_precomputed is not None:
        # Validate
        from fractions import Fraction
        total = sum(Fraction(1, d) for d in _best_precomputed)
        if total == 1 and len(set(_best_precomputed)) == len(_best_precomputed):
            if all(5 <= d <= 10000000 for d in _best_precomputed):
                if 1 <= len(_best_precomputed) <= 256:
                    return _best_precomputed
    
    # Fallback: try to find a solution quickly
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s limit
    
    # Try 4-term search with more range
    from fractions import Fraction
    for a in range(5, 15):
        if time.time() > deadline:
            break
        for b in range(a+1, 80):
            s2 = Fraction(1,1) - Fraction(1,a) - Fraction(1,b)
            if s2 <= 0:
                break
            for c in range(b+1, 300):
                s3 = s2 - Fraction(1,c)
                if s3 <= 0:
                    break
                if s3.numerator == 1:
                    d = s3.denominator
                    if d > c and d <= 10000000:
                        result = [a, b, c, d]
                        # Validate
                        total = sum(Fraction(1, x) for x in result)
                        if total == 1:
                            return result
                elif s3.denominator % s3.numerator == 0:
                    d = s3.denominator // s3.numerator
                    if d > c and d <= 10000000:
                        result = [a, b, c, d]
                        total = sum(Fraction(1, x) for x in result)
                        if total == 1:
                            return result
    
    # Try 5-term search
    for a in range(5, 12):
        if time.time() > deadline:
            break
        for b in range(a+1, 30):
            s2 = Fraction(1,1) - Fraction(1,a) - Fraction(1,b)
            if s2 <= 0:
                break
            for c in range(b+1, 80):
                s3 = s2 - Fraction(1,c)
                if s3 <= 0:
                    break
                for d in range(c+1, 150):
                    if time.time() > deadline:
                        break
                    s4 = s3 - Fraction(1,d)
                    if s4 <= 0:
                        break
                    if s4.numerator == 1:
                        e = s4.denominator
                        if e > d and e <= 10000000:
                            result = [a, b, c, d, e]
                            total = sum(Fraction(1, x) for x in result)
                            if total == 1:
                                return result
                    elif s4.denominator % s4.numerator == 0:
                        e = s4.denominator // s4.numerator
                        if e > d and e <= 10000000:
                            result = [a, b, c, d, e]
                            total = sum(Fraction(1, x) for x in result)
                            if total == 1:
                                return result
    
    # Last resort: use a known valid solution (verify at runtime)
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # This sums to more than 1, so pick a subset that equals 1... hard to guarantee.
    # Use the original seed approach as ultimate fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total_val, values in list(reached.items()):
            if total_val + w <= target and total_val + w not in reached:
                reached[total_val + w] = values + [d]
        if target in reached:
            result = reached[target]
            if len(result) <= 256:
                return result
        if time.time() > deadline:
            break
    
    raise RuntimeError("no solution found within time limit")

# EVOLVE-BLOCK-END
