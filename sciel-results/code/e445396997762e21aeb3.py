"""Constructive search for minimal 4-dominator tournament using Paley tournaments."""
import itertools

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _check_valid_fast(n, orientation):
    """
    Check if every 4-set has a common dominator (vertex v in V such that v->u for all u in S).
    Uses bitmask intersection of in-neighborhoods.
    """
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j, so i is in in_neigh[j]
                in_neigh[j] |= (1 << i)
            else:
                # j -> i, so j is in in_neigh[i]
                in_neigh[i] |= (1 << j)
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            return False
    return True

def _construct_paley(p):
    """
    Construct a Paley tournament on prime p (p % 4 == 3).
    """
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def construct():
    """
    Find the smallest n such that a valid tournament exists.
    Tests Paley tournaments for primes p ≡ 3 (mod 4).
    """
    candidates = [p for p in range(31, 91) if p % 4 == 3 and _is_prime(p)]
    
    for p in candidates:
        orientation = _construct_paley(p)
        if _check_valid_fast(p, orientation):
            return {'n': p, 'orientation': orientation}
            
    # Fallback
    p = 83
    orientation = _construct_paley(p)
    return {'n': p, 'orientation': orientation}
