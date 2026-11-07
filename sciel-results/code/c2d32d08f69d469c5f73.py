# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _can_add(a, S, S_arr):
    """
    Check if integer a can be added to set S (represented as list S and numpy array S_arr).
    Conditions:
    1. For all distinct b, c in S, a does not divide b*c.
    2. For all distinct c, d in S, c does not divide a*d.
    3. For all distinct b, d in S, b does not divide a*d. (Symmetric to 2)
    """
    n = len(S)
    if n < 2:
        return True
    
    # Check 1: a | b*c for distinct b,c in S
    # Compute outer product modulo a
    # We need to check off-diagonal elements
    prods = np.outer(S_arr, S_arr)
    divs = prods % a
    # Mask the diagonal (b == c)
    # divs is n x n. Diagonal indices are 0, n+1, 2n+2, ...
    mask = np.ones((n, n), dtype=bool)
    mask.flat[::n+1] = False
    if np.any(divs[mask] == 0):
        return False
        
    # Check 2 & 3: b | a*c for distinct b,c in S
    # This is equivalent to checking if there exist distinct b,c in S such that (a*c) % b == 0
    # Let's compute (a * S_arr) % S_arr
    # Result is n x n matrix where M[i, j] = (a * S[j]) % S[i]
    # We need to check if any M[i, j] == 0 for i != j
    # Note: S[i] is b, S[j] is c.
    # Condition: b does not divide a*c.
    vals = a * S_arr
    divs2 = vals[:, None] % S_arr[None, :]
    # Mask diagonal
    if np.any(divs2[mask] == 0):
        return False
        
    return True

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Start with all primes in [2, 256]
    primes = [x for x in range(2, 257) if _is_prime(x)]
    S = list(primes)
    S_arr = np.array(S, dtype=np.int64)
    
    composites = [x for x in range(2, 257) if not _is_prime(x)]
    
    # Sort composites by number of prime factors (ascending) to add "simpler" ones first
    # This heuristic often works better for packing problems
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
            
    composites.sort(key=prime_factor_count)
    
    # Greedy addition
    for a in composites:
        if time.time() > deadline - 2.0:
            break
            
        if _can_add(a, S, S_arr):
            S.append(a)
            S_arr = np.array(S, dtype=np.int64)
            
    # Optional: Local search / Hill climbing
    # Try to swap out a composite for a different one if it allows adding more?
    # Or just try to add remaining composites in a different order?
    # Given the small size, we might be able to do a more thorough check.
    
    # Let's try a second pass with composites sorted by value descending?
    # Or just ensure we didn't miss any due to order.
    # The greedy choice is path-dependent.
    # Let's try to see if we can add any more by re-evaluating?
    # Actually, if a composite was rejected, adding other numbers won't make it valid (constraints are monotonic: if a|bc for some b,c in S, adding more elements to S won't remove b,c, so it remains invalid).
    # So greedy is "safe" in that it never invalidates the set, but it might block future additions.
    # However, since we check all composites, and the set only grows, if a number was rejected, it will remain rejected.
    # So the order only matters for *which* numbers get added.
    # To maximize cardinality, we want to add numbers that are "least restrictive".
    # Numbers with more prime factors are more restrictive?
    # If a = p*q, it blocks b,c where p|b, q|c.
    # If a = p*q*r, it blocks more combinations.
    # So adding numbers with FEWER prime factors first is a good heuristic.
    
    # Let's try to improve the set by removing some composites and adding others?
    # This is complex. Given the 60s limit and small N, the current greedy with NumPy should be fast enough.
    # Let's just return the result.
    
    result = [int(x) for x in S]
    return result
# EVOLVE-BLOCK-END
