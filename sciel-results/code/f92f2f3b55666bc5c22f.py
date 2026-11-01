"""Kissing number in dimension 11 — constructive D11 shortest vectors + W1."""
import numpy as np
import itertools
import time

D = 11


def construct():
    # All shortest vectors of D11 have squared norm 4 and come in two families:
    #   Type A: ±e_i ± e_j  (weight 2, entries in {-1,0,1}, norm²=2 → actually norm²=2)
    #   Type B: (1/2)(±1,±1,...,±1) with product of signs = +1 (weight 11, entries ±1/2)
    #
    # But the task requires INTEGER vectors. The D11 lattice in its standard
    # embedding has vectors like e_i - e_j (integers) and also half-integer
    # vectors. To get integer vectors with the kissing property, we can:
    #
    # Strategy: Use the 220 vectors of type e_i - e_j (i<j) and e_i + e_j (i≠j).
    #   - e_i - e_j: norm² = 2
    #   - e_i + e_j (i≠j): norm² = 2
    #   Wait, that's 11*10 = 110 (for i≠j, e_i±e_j gives 2 per pair = 11*10=110)
    #   Actually for each unordered pair {i,j}, i≠j, we have 4 vectors:
    #     e_i - e_j, e_j - e_i, e_i + e_j, -(e_i + e_j)
    #   That's C(11,2)*4 = 55*4 = 220 vectors, all with norm² = 2.
    #
    # Check mutual distances:
    #   For two such vectors u, v with norm²=2:
    #   ||u - v||² = ||u||² + ||v||² - 2<u,v> = 4 - 2<u,v>
    #   We need ||u-v|| >= max_norm = sqrt(2), i.e., ||u-v||² >= 2, i.e., 4 - 2<u,v> >= 2
    #   => <u,v> <= 1.
    #
    #   Possible dot products between distinct vectors of this set:
    #   - Same vector: dot=2 (excluded, distinct)
    #   - u = e_i - e_j, v = e_i - e_k (j≠k): dot = 1
    #   - u = e_i - e_j, v = e_j - e_k (i≠k, k≠i): dot = -1
    #   - u = e_i - e_j, v = e_i + e_j: dot = 0
    #   - u = e_i - e_j, v = e_k + e_l (disjoint): dot = 0
    #   - u = e_i - e_j, v = e_j + e_k (k≠i): dot = -1
    #   - u = e_i + e_j, v = e_k + e_l (disjoint): dot = 0
    #   - u = e_i + e_j, v = e_i + e_k (j≠k): dot = 1
    #   - u = e_i + e_j, v = e_i - e_k (k≠j): dot = 1
    #   - u = e_i + e_j, v = e_j - e_k (k≠i): dot = 1
    #
    #   Max dot product between distinct vectors is 1. So condition <u,v> <= 1 is satisfied.
    #   Therefore all 220 vectors are mutually compatible!
    #
    # Additionally, we can add W1 vectors: (2,0,...,0) etc.
    #   W1 = 2*e_i (or -2*e_i), norm² = 4.
    #   Dot with e_i ± e_j: 2*1 = 2 or 2*(-1) = -2. Max |dot| = 2.
    #   Need ||W1 - v||² >= max_norm² = 4 (since max norm is now 2 from W1 vectors).
    #   ||2e_i - (e_i ± e_j)||² = 4 + 2 ∓ 4 = 6 or 0.
    #   Wait: 2e_i - e_i - e_j = e_i - e_j, norm² = 2 < 4. CONFLICT!
    #   2e_i - e_i + e_j = e_i + e_j, norm² = 2 < 4. CONFLICT!
    #   So W1 vectors conflict with the 220 vectors.
    #
    #   What about W1 vectors among themselves?
    #   2e_i and 2e_j (i≠j): ||2e_i - 2e_j||² = 8 >= 4. OK.
    #   2e_i and -2e_i: ||4e_i||² = 16 >= 4. OK.
    #   So all 22 W1 vectors are mutually compatible, but conflict with the 220.
    #
    #   Best choice: 220 vectors of norm² = 2.
    #
    # Can we do better? The kissing number of R^11 is known to be at least 556.
    # But with integer vectors, the D11 construction gives 220 + possibly more.
    #
    # Let's also consider the 11 vectors of type (1,1,...,1) scaled? No, those aren't integers in the right way.
    #
    # Actually, let's reconsider. The Leech lattice in 24D has kissing number 196560.
    # In 11D, the best known kissing number is 556 (from 2003, by Nebe and Sloane).
    #
    # But for INTEGER vectors, let's think about what's achievable.
    #
    # Alternative: Use vectors from the root lattice D11:
    #   D11 = {x in Z^11 : sum(x_i) is even}
    #   Shortest vectors have norm² = 2: e_i ± e_j
    #   Count: 11*10 = 110 (for i≠j, each pair gives e_i-e_j and e_i+e_j, but e_i-e_j for i≠j is 11*10=110, and e_i+e_j for i≠j is also 11*10=110... wait)
    #
    #   Actually in D11, the vectors of norm² = 2 are e_i - e_j for i≠j (that's 11*10 = 110)
    #   and e_i + e_j for i≠j where sum is even... e_i + e_j has sum = 2 which is even, so all are in D11.
    #   So 110 + 110 = 220 vectors. Yes.
    #
    # These 220 form a valid kissing configuration with norm² = 2.
    #
    # Can we add more? What about vectors of norm² = 4 in D11?
    #   These include 2e_i (weight 1, sum=2 even, in D11) — but we saw they conflict.
    #   Also e_i + e_j + e_k + e_l (weight 4, sum=4 even) — norm² = 4.
    #   Check dot with e_i ± e_j: can be 2, which would give distance² = 4+2-2*2 = 2 < 4. Conflict.
    #
    #   So we can't mix norm²=2 and norm²=4 vectors easily.
    #
    # Let's just return the 220 vectors of norm² = 2.

    vectors = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            # e_i - e_j
            v1 = [0] * D
            v1[i] = 1
            v1[j] = -1
            vectors.append(v1)
            # e_i + e_j
            v2 = [0] * D
            v2[i] = 1
            v2[j] = 1
            vectors.append(v2)

    result = np.array(vectors, dtype=np.int64)

    # Verify: all nonzero, min pairwise distance >= max norm
    norms_sq = np.sum(result ** 2, axis=1)
    max_norm_sq = norms_sq.max()
    # Check all pairwise distances
    # For efficiency, check via dot products: ||u-v||² = ||u||²+||v||²-2<u,v>
    # Since all have same norm², ||u-v||² = 2*max_norm_sq - 2<u,v>
    # Need 2*max_norm_sq - 2<u,v> >= max_norm_sq  =>  max_norm_sq >= 2<u,v>  =>  <u,v> <= max_norm_sq/2
    dots = result @ result.T
    np.fill_diagonal(dots, 0)
    max_dot_offdiag = dots.max()
    # Condition: max_dot_offdiag <= max_norm_sq / 2
    assert max_dot_offdiag <= max_norm_sq // 2, f"Validation failed: max_dot={max_dot_offdiag}, max_norm_sq={max_norm_sq}"

    return result
