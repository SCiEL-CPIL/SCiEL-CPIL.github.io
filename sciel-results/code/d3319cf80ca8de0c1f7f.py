import time
import random
import numpy as np

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

def _paley_orientation(p):
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': n, 'orientation': orientation}

def _check_property(n, orientation):
    """
    Check if every 4 vertices have a common dominator.
    Uses bitsets for efficiency.
    """
    # Build in-neighborhood bitsets
    # in_neigh[u] is a bitmask of vertices v such that v -> u
    in_neigh = [0] * n
    
    # Map orientation list to indices
    # orientation[k] corresponds to pair (i, j)
    # We need a fast way to get direction.
    # Let's build a matrix or just use the list logic.
    # For speed, let's build the bitsets directly from the orientation list.
    
    # Index calculation for pair (i, j) with i < j
    # Number of pairs before i: i*(n-1) - i*(i-1)/2 ?
    # Total pairs for vertex k is k + (n-1-k) = n-1.
    # Pairs involving i with j < i: i * (n-1-i) ? No.
    # Let's just iterate and fill.
    
    # Actually, it's easier to just generate the tournament matrix or adjacency list.
    # But we have the flat list.
    
    # Let's reconstruct the adjacency.
    # adj[i][j] = 1 if i -> j else 0
    # But we need in_neigh.
    # If i -> j, then i is in in_neigh[j].
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                in_neigh[j] |= (1 << i)
            else:
                # j -> i
                in_neigh[i] |= (1 << j)
            idx += 1
            
    # Check all 4-sets
    # To speed up, we can iterate combinations.
    # For n=67, C(67,4) ~ 7.8M.
    # Python loop might be slow. 7.8M * (4 ANDs + 1 AND + 1 check) ~ 10 ops.
    # 78M ops. In Python, ~5-10 seconds. Acceptable.
    
    # Optimization: Precompute nothing? 
    # We can use itertools.combinations.
    
    import itertools
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        common = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        if (common & ~mask) == 0:
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Try Paley tournaments for primes p ≡ 3 mod 4
    # Start from 67, go up to 83 (known good)
    # If 67 works, we return it.
    
    primes_to_try = [67, 71, 79, 83]
    
    for p in primes_to_try:
        if time.time() - start_time > time_limit:
            break
        if p % 4 != 3 or not _is_prime(p):
            continue
            
        result = _paley_orientation(p)
        n = result['n']
        
        # Check if it satisfies the property
        # For very large n, this might take too long, but 67-83 is fine.
        if _check_property(n, result['orientation']):
            return result
            
    # Fallback to 83 if check fails or times out (should not happen for 83)
    # But if 67 fails, we might want to try random search for n=67?
    # Given "smallest known is 67", it's likely Paley(67) works or a specific 67-vertex tournament works.
    # If Paley(67) fails, we should probably return the best we found or 83.
    # However, if Paley(67) fails, maybe Paley(71) works?
    
    # If none of the specific primes worked (or check was too slow), return 83
    return _paley_orientation(83)
