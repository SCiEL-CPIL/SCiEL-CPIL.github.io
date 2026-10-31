"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _is_cap_set(arr):
    """Check if the set of vectors (rows of arr) is a cap set in F_3^n."""
    n = arr.shape[1]
    m = arr.shape[0]
    if m < 3:
        return True
    # A set is a cap set iff no three distinct points are collinear in F_3^n.
    # In F_3, three points a, b, c are collinear iff a + b + c = 0 (mod 3)
    # with a, b, c distinct.
    # We can check this by: for each pair (i, j), compute a = arr[i] + arr[j] (mod 3),
    # and check if a is in the set and a is distinct from arr[i] and arr[j].
    # More efficiently: a cap set has no solution to x + y + z = 0 with x, y, z distinct.
    # Equivalently, no x, y with x != y such that x + y is in the set and x + y != x and x + y != y.
    # In F_3, x + y = x implies y = 0, x + y = y implies x = 0. So we just need:
    # no two distinct vectors whose sum (mod 3) is also in the set.
    
    # Use a set of tuples for fast lookup
    vec_set = set(map(tuple, arr))
    for i in range(m):
        for j in range(i + 1, m):
            s = (arr[i] + arr[j]) % 3
            if tuple(s) in vec_set:
                # Check distinctness: s != arr[i] and s != arr[j]
                # In F_3: s == arr[i] iff arr[j] == 0, s == arr[j] iff arr[i] == 0
                if not np.array_equal(s, arr[i]) and not np.array_equal(s, arr[j]):
                    return False
    return True

def _is_cap_set_fast(arr):
    """Faster cap set check using set operations."""
    m = arr.shape[0]
    if m < 3:
        return True
    vec_set = set(map(tuple, arr))
    for i in range(m):
        ai = arr[i]
        for j in range(i + 1, m):
            s = (ai + arr[j]) % 3
            ts = tuple(s)
            if ts in vec_set:
                # Check that s is distinct from both ai and arr[j]
                if not np.array_equal(s, ai) and not np.array_equal(s, arr[j]):
                    return False
    return True

def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    return 0.0

# EVOLVE-BLOCK-END
