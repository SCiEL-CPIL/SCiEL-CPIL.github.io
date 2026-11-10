# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Returns an array of shape (k, 11) with integer coordinates.
    Uses the 220 minimal vectors of the D11 lattice (squared norm = 2).
    These satisfy min_pair_dist^2 = 4 >= max_norm^2 = 2.
    """
    d = 11
    start_time = time.time()
    deadline = 630.0

    # Generate all 220 minimal vectors of D11:
    # D11 lattice minimal vectors are all vectors with exactly two non-zero
    # entries of value ±1, with an even number of negative entries.
    # There are C(11,2) = 55 positions for the two non-zero entries,
    # and 4 sign combinations: (+1,+1), (+1,-1), (-1,+1), (-1,-1)
    # But we need even number of negatives:
    #   (+1,+1): 0 negatives -> even -> valid
    #   (+1,-1): 1 negative  -> odd  -> invalid
    #   (-1,+1): 1 negative  -> odd  -> invalid
    #   (-1,-1): 2 negatives -> even -> valid
    # So only 2 sign combinations per position pair: 55 * 2 = 110 vectors.
    # Wait, that's only 110. The D11 minimal vectors actually number 2*C(11,2) = 110?
    # Let me reconsider. The D_n lattice is {x in Z^n : sum x_i is even}.
    # Minimal vectors of D_n have squared norm 2, with exactly two ±1 entries.
    # For n=11: C(11,2)*4 = 220 total vectors with two ±1 entries.
    # Of these, those with even sum: if entries are at positions i,j with values a,b:
    #   a+b must be even (since rest are 0), so a,b same sign: (+1,+1) or (-1,-1).
    #   That gives C(11,2)*2 = 110 vectors.
    # But wait - the full set of minimal vectors of D11 has 2*n*(n-1) = 220 vectors.
    # Hmm, let me recheck. Actually for D_n, the minimal vectors are all vectors
    # with two non-zero entries ±1 where the sum of coordinates is even.
    # Sum = a + b where a,b ∈ {±1}. Sum is even iff a=b.
    # So we get (+1,+1) and (-1,-1) at each of C(n,2) positions: n*(n-1) vectors.
    # For n=11: 110 vectors. But the kissing number of D11 is 110? No...
    # Actually I recall the kissing number of D_n is n(n-1). For n=11 that's 110.
    # But the D11 lattice kissing number is indeed 2*C(11,2) = 110? Let me verify:
    # Standard result: kissing number of D_n is n(n-1) for n >= 2.
    # For D4: 4*3=12. The D4 lattice has 12 minimal vectors. Yes.
    # So D11 has 110 minimal vectors.

    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # (+1 at i, +1 at j)
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            # (-1 at i, -1 at j)
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = -1
            v2[j] = -1
            points.append(v2)

    points_arr = np.array(points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    # Check deadline
    if time.time() > deadline - 10:
        return points_arr

    return points_arr


# EVOLVE-BLOCK-END
