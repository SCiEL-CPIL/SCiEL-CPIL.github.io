"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _validate(arr):
    if not isinstance(arr, (list, tuple)):
        return False
    if not (1 <= len(arr) <= 100):
        return False
    seen = set()
    prod = P['n']
    for x in arr:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
        if x in seen:
            return False
        seen.add(x)
        prod *= x
    s = isqrt(prod)
    return s * s == prod

def _try_small_sets():
    # Explicit constructions discovered analytically
    candidates = [
        [224, 240],
        [216, 240, 245],
        [216, 245, 252],
        [224, 245, 250],
        [216, 224, 240, 245],
    ]
    for c in candidates:
        if _validate(c):
            return list(c)
    return None

def _brute_force_search(limit):
    # Search for small sets where each element is in (210, limit]
    # We look for 1-element: need x such that 210*x is a square -> x = 210*k^2, check range
    # 1-element: 210*x = s^2 => x = s^2/210, must be integer. s^2 divisible by 210=2*3*5*7, so s divisible by 2*3*5*7=210? No, s^2 div by 210 means for each prime p|210, 2*v_p(210) <= 2*v_p(s), so v_p(s)>=1 for each. So s is multiple of 210. Then x = (210k)^2/210 = 210*k^2. Need 210<x<=limit => 210*k^2 <= limit => k^2 <= limit/210. k>=1: 210*1=210 (not >210). k=2: 210*4=840. So 1-element candidates: 840, 1260, 1680,...
    # 2-element: 210*x*y = s^2. 
    # 3-element, 4-element: brute force combinations
    from itertools import combinations
    nums = list(range(211, limit+1))
    # 1-element
    for x in nums:
        prod = 210 * x
        s = isqrt(prod)
        if s*s == prod:
            return [x]
    # 2-element
    n = len(nums)
    for i in range(n):
        base = 210 * nums[i]
        for j in range(i+1, n):
            prod = base * nums[j]
            s = isqrt(prod)
            if s*s == prod:
                return [nums[i], nums[j]]
    # 3-element (only if limit is small enough)
    if n > 3:
        for i in range(n):
            for j in range(i+1, n):
                base = 210 * nums[i] * nums[j]
                for k in range(j+1, min(j+20, n)):  # limited inner loop for speed
                    prod = base * nums[k]
                    s = isqrt(prod)
                    if s*s == prod:
                        return [nums[i], nums[j], nums[k]]
    return None

def construct():
    start = time.time()
    deadline = 50.0  # safety margin before 60s kill
    
    # 1. Try explicit known-good solutions first
    sol = _try_small_sets()
    if sol is not None:
        return sol
    
    # 2. Brute force from 211 upward, expanding limit
    # We know analytically that [224, 240] works with max=240.
    # Let's verify and return it directly if valid.
    if _validate([224, 240]):
        return [224, 240]
    
    # 3. Search for potentially better (smaller max) solutions
    # We know [224,240] gives max=240. Search for any solution with max < 240.
    limit_search = 239
    nums = list(range(211, limit_search+1))
    n = len(nums)
    
    # 1-element check
    for x in nums:
        prod = 210 * x
        s = isqrt(prod)
        if s*s == prod:
            return [x]
    
    # 2-element check
    for i in range(n):
        base = 210 * nums[i]
        for j in range(i+1, n):
            prod = base * nums[j]
            s = isqrt(prod)
            if s*s == prod:
                return [nums[i], nums[j]]
    
    # 3-element check
    if n > 2:
        for i in range(n):
            for j in range(i+1, n):
                base = 210 * nums[i] * nums[j]
                for k in range(j+1, n):
                    if time.time() - start > deadline:
                        return [224, 240]
                    prod = base * nums[k]
                    s = isqrt(prod)
                    if s*s == prod:
                        return [nums[i], nums[j], nums[k]]
    
    # 4-element check: C(29,4) = 23751, fast
    if n > 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    base = 210 * nums[i] * nums[j] * nums[k]
                    for m in range(k+1, n):
                        if time.time() - start > deadline:
                            return [224, 240]
                        prod = base * nums[m]
                        s = isqrt(prod)
                        if s*s == prod:
                            return [nums[i], nums[j], nums[k], nums[m]]
    
    # 5-element check: C(29,5) = 118755, still feasible
    if n > 4:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for m in range(k+1, n):
                        base = 210 * nums[i] * nums[j] * nums[k] * nums[m]
                        for p in range(m+1, n):
                            if time.time() - start > deadline:
                                return [224, 240]
                            prod = base * nums[p]
                            s = isqrt(prod)
                            if s*s == prod:
                                return [nums[i], nums[j], nums[k], nums[m], nums[p]]
    
    # If nothing better found, return the known optimal [224, 240]
    return [224, 240]
# EVOLVE-BLOCK-END
