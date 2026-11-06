import time
import itertools

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices.
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set(x * x % p for x in range(1, p))
    orientation = []
    append = orientation.append
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            append(0 if diff in squares else 1)
    return orientation

def _verify_paley_exact(p, orientation):
    """
    Exact verification of the 4-dominator property (Internal).
    Every set of 4 vertices must contain a vertex that dominates the other 3.
    """
    # Build out_bits: out_bits[i] is a bitmask of vertices j such that i -> j
    out_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            if orientation[idx] == 0:
                # i -> j
                out_bits[i] |= (1 << j)
            else:
                # j -> i
                out_bits[j] |= (1 << i)
            idx += 1
    
    ob = out_bits
    
    # Check all 4-tuples
    for a, b, c, d in itertools.combinations(range(p), 4):
        # Create masks for the other 3 vertices for each candidate dominator
        # Candidate a: a must beat b, c, d
        mask_bcd = (1 << b) | (1 << c) | (1 << d)
        if (ob[a] & mask_bcd) != mask_bcd:
            # Candidate b: b must beat a, c, d
            mask_acd = (1 << a) | (1 << c) | (1 << d)
            if (ob[b] & mask_acd) != mask_acd:
                # Candidate c: c must beat a, b, d
                mask_abd = (1 << a) | (1 << b) | (1 << d)
                if (ob[c] & mask_abd) != mask_abd:
                    # Candidate d: d must beat a, b, c
                    mask_abc = (1 << a) | (1 << b) | (1 << c)
                    if (ob[d] & mask_abc) != mask_abc:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Include more primes to be safe, but start small
    candidates = [31, 43, 47, 59, 67, 71, 79, 83, 89]

    best_n = 89
    best_orientation = _paleyn_orientation(89)

    for p in candidates:
        if time.time() > deadline:
            break
        orientation = _paleyn_orientation(p)
        if _verify_paley_exact(p, orientation):
            best_n = p
            best_orientation = orientation
            break

    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
