"""Program for large reciprocal-subsum-free sets using optimized greedy construction."""
import time
import math
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _check_validity(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each element x in S, 1/x must NOT equal the sum of reciprocals of any subset of S\{x}.
    """
    if not S:
        return True
    
    # Convert to reciprocals
    reciprocals = {x: 1.0 / x for x in S}
    
    for x in S:
        target = reciprocals[x]
        others = [reciprocals[y] for y in S if y != x]
        
        # Check all subsets of 'others'
        n = len(others)
        for mask in range(1, 1 << n):
            total = 0.0
            for i in range(n):
                if mask & (1 << i):
                    total += others[i]
            # Use a small tolerance for floating point comparison
            if abs(total - target) < 1e-12:
                return False
    return True


def _check_validity_exact(S):
    """
    Exact rational check for validity.
    For each x in S, check if 1/x = sum(1/y for y in subset of S\{x}).
    This is equivalent to: 1/x = sum(1/y)
    Multiply by LCM of all elements in S to avoid floating point errors.
    """
    if not S:
        return True
    
    # Compute LCM of all elements in S
    L = 1
    for x in S:
        L = _lcm(L, x)
    
    # Convert reciprocals to integers: 1/x becomes L/x
    # Condition: L/x == sum(L/y for y in subset)
    # => 1/x == sum(1/y)
    
    for x in S:
        target_num = L // x  # This is the numerator when denominator is L
        others = [y for y in S if y != x]
        n = len(others)
        
        # Check all subsets of 'others'
        for mask in range(1, 1 << n):
            total_num = 0
            for i in range(n):
                if mask & (1 << i):
                    total_num += L // others[i]
            if total_num == target_num:
                return False
    return True


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses greedy descending construction with exact feasibility checks.
    """
    if _time_remaining() <= 0:
        return list(range(13, 25))
    
    # Try greedy descending: start with largest numbers, add if valid
    S = []
    for x in range(24, 0, -1):
        if _time_remaining() <= 1.0:
            break
        candidate = S + [x]
        if _check_validity_exact(candidate):
            S.append(x)
    
    if len(S) > 0:
        return S
    
    # Fallback
    return list(range(13, 25))


# Test the greedy approach
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Cardinality: {len(result)}")
    print(f"Valid: {_check_validity_exact(result)}")
