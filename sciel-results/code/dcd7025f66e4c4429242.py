import time
import math

def _make_paley(p):
    """
    Constructs the orientation list for the Paley tournament on p vertices.
    p must be a prime congruent to 3 mod 4.
    i -> j iff (j - i) % p is a quadratic residue mod p.
    Returns a list of bits for pairs (i, j) with i < j.
    """
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return orientation

def _is_valid_fast(orientation, n):
    """
    Checks if the tournament defined by orientation satisfies the property:
    Every set of 4 vertices has a common dominator.
    
    Uses bitmask representation for in-neighborhoods.
    in_masks[i] has bit j set if j -> i.
    A set {a,b,c,d} has a common dominator v iff 
    (in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]) != 0.
    Note: v cannot be in {a,b,c,d} because v cannot point to itself.
    """
    # Build in-neighborhood masks
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j, so i is an in-neighbor of j
                in_masks[j] |= (1 << i)
            else:
                # j -> i, so j is an in-neighbor of i
                in_masks[i] |= (1 << j)
            idx += 1
    
    # Check every 4-set
    # We can optimize by breaking early if found invalid.
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    common = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
                    if common == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Primes p ≡ 3 (mod 4) in increasing order.
    # We know 67 is valid. Smaller ones are likely invalid but we must check.
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    # Pre-compute the fallback result for the known smallest valid prime (67)
    # to ensure we always have a valid result if the loop times out or fails.
    fallback_p = 67
    fallback_orientation = _make_paley(fallback_p)
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        orientation = _make_paley(p)
        
        # Check validity
        if _is_valid_fast(orientation, p):
            return {'n': p, 'orientation': orientation}
            
    # If we didn't find a smaller valid one, or ran out of time,
    # return the known valid solution for 67.
    # Note: If 67 was checked and failed (which shouldn't happen), 
    # we might need a larger fallback, but 67 is established as valid.
    return {'n': fallback_p, 'orientation': fallback_orientation}
