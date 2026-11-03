"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = n * m * p  # 75 schoolbook

    # We'll build the rank-1 terms as lists of (U_col, V_col, W_col)
    # Each term is a rank-1 tensor: u ⊗ v ⊗ w where:
    #   u has length n*m, v has length m*p, w has length p*n
    #   T[i,j,k] = sum_c u[i,c]*v[j,c]*w[k,c]
    # But our convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # Schoolbook terms
    terms = []  # list of (u_dict, v_dict, w_dict) where dicts map index->coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))

    # Now try to replace some schoolbook terms with Strassen-like combinations
    # For a 2x2x2 block at (i0..i0+1, j0..j0+1, k0..k0+1):
    # Instead of 8 terms, use 7 Strassen terms.
    # The Strassen decomposition for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form, each Mk contributes:
    # u vector (coefficients on A entries), v vector (coefficients on B entries), w vector (coefficients on C entries)

    def strassen_222_terms(i0, j0, k0):
        """Return 7 rank-1 terms for the 2x2x2 block starting at (i0,j0,k0)."""
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        # C indices: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)

        A00 = (i0, j0)
        A01 = (i0, j0+1)
        A10 = (i0+1, j0)
        A11 = (i0+1, j0+1)

        B00 = (j0, k0)
        B01 = (j0, k0+1)
        B10 = (j0+1, k0)
        B11 = (j0+1, k0+1)

        C00 = (k0, i0)
        C01 = (k0, i0+1)
        C10 = (k0+1, i0)
        C11 = (k0+1, i0+1)

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        t1_u = {A00: 1, A11: 1}
        t1_v = {B00: 1, B11: 1}
        t1_w = {C00: 1, C11: 1}

        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        t2_u = {A10: 1, A11: 1}
        t2_v = {B00: 1}
        t2_w = {C10: 1, C11: -1}

        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        t3_u = {A00: 1}
        t3_v = {B01: 1, B11: -1}
        t3_w = {C01: 1, C11: 1}

        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        t4_u = {A11: 1}
        t4_v = {B10: 1, B00: -1}
        t4_w = {C00: 1, C10: 1}

        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        t5_u = {A00: 1, A01: 1}
        t5_v = {B11: 1}
        t5_w = {C00: -1, C01: 1}

        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        t6_u = {A10: 1, A00: -1}
        t6_v = {B00: 1, B01: 1}
        t6_w = {C11: 1}

        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        t7_u = {A01: 1, A11: -1}
        t7_v = {B10: 1, B11: 1}
        t7_w = {C00: 1}

        return [(t1_u, t1_v, t1_w),
                (t2_u, t2_v, t2_w),
                (t3_u, t3_v, t3_w),
                (t4_u, t4_v, t4_w),
                (t5_u, t5_v, t5_w),
                (t6_u, t6_v, t6_w),
                (t7_u, t7_v, t7_w)]

    # Strategy: find 2x2x2 blocks in the 3x5x5 tensor and replace their 8 schoolbook
    # terms with 7 Strassen terms. We can apply Strassen to non-overlapping 2x2x2 blocks.
    # The tensor has indices i in [0,2], j in [0,4], k in [0,4].
    # A 2x2x2 block uses 2 consecutive i, 2 consecutive j, 2 consecutive k values.
    # Possible i ranges: [0,1] or [1,2]
    # Possible j ranges: [0,1], [1,2], [2,3], [3,4]
    # Possible k ranges: [0,1], [1,2], [2,3], [3,4]

    # Greedy approach: try all possible sets of non-overlapping 2x2x2 blocks
    # and pick the one that maximizes the number of blocks (each saves 1 rank).

    # For a given set of blocks, we need to:
    # 1. Remove the 8 schoolbook terms for each block's indices
    # 2. Add the 7 Strassen terms

    # Let's enumerate all maximal sets of non-overlapping 2x2x2 blocks.
    # A block is identified by (i0, j0, k0) where it covers:
    #   i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # Two blocks overlap if they share any (i,j,k) index triple.

    all_blocks = []
    for i0 in range(2):  # 0 or 1
        for j0 in range(4):  # 0..3
            for k0 in range(4):  # 0..3
                all_blocks.append((i0, j0, k0))

    # Check overlap: blocks (i0a,j0a,k0a) and (i0b,j0b,k0b) overlap if
    # {i0a, i0a+1} ∩ {i0b, i0b+1} != empty AND {j0a, j0a+1} ∩ {j0b, j0b+1} != empty AND {k0a, k0a+1} ∩ {k0b, k0b+1} != empty
    def overlap(b1, b2):
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        if not ({i0a, i0a+1} & {i0b, i0b+1}):
            return False
        if not ({j0a, j0a+1} & {j0b, j0b+1}):
            return False
        if not ({k0a, k0a+1} & {k0b, k0b+1}):
            return False
        return True

    # Find all maximal independent sets (or just greedily find large ones)
    # Since there are 2*4*4 = 32 blocks, we can try greedy with different orderings.

    best_rank = rank  # start with schoolbook
    best_terms = None

    # Greedy: sort blocks, add if no overlap with already selected
    # Try multiple orderings
    import random
    rng = random.Random(42)

    candidates = []
    # Try several random orderings
    for trial in range(50):
        if time.time() > DEADLINE:
            break
        shuffled = all_blocks[:]
        rng.shuffle(shuffled)
        selected = []
        for b in shuffled:
            if all(not overlap(b, s) for s in selected):
                selected.append(b)
        if len(selected) > 0:
            candidates.append(selected)

    # Also try systematic orderings
    for order in ['ij', 'jk', 'ki', 'ij_kj']:
        if time.time() > DEADLINE:
            break
        if order == 'ij':
            sorted_blocks = sorted(all_blocks, key=lambda b: (b[0], b[1], b[2]))
        elif order == 'jk':
            sorted_blocks = sorted(all_blocks, key=lambda b: (b[1], b[2], b[0]))
        elif order == 'ki':
            sorted_blocks = sorted(all_blocks, key=lambda b: (b[2], b[0], b[1]))
        else:
            sorted_blocks = sorted(all_blocks, key=lambda b: (b[0]+b[1], b[1]+b[2], b[0]+b[2]))
        selected = []
        for b in sorted_blocks:
            if all(not overlap(b, s) for s in selected):
                selected.append(b)
        if len(selected) > 0:
            candidates.append(selected)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = frozenset(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    # For each candidate, compute the resulting rank
    for selected in unique_candidates:
        if time.time() > DEADLINE:
            break
        # Number of schoolbook terms removed: 8 per block
        # Number of Strassen terms added: 7 per block
        new_rank = rank - 8 * len(selected) + 7 * len(selected)
        new_rank = rank - len(selected)  # net saving of 1 per block

        if new_rank < best_rank:
            best_rank = new_rank
            best_terms = selected

    # Now construct the final decomposition
    if best_terms is not None and best_rank < rank:
        # Build the term list
        final_terms = []

        # Determine which schoolbook terms are "covered" by Strassen blocks
        covered = set()
        for (i0, j0, k0) in best_terms:
            for i in range(i0, i0+2):
                for j in range(j0, j0+2):
                    for k in range(k0, k0+2):
                        covered.add((i, j, k))

        # Add non-covered schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        final_terms.append((u, v, w))

        # Add Strassen terms
        for (i0, j0, k0) in best_terms:
            for term in strassen_222_terms(i0, j0, k0):
                final_terms.append(term)

        rank_out = len(final_terms)
    else:
        final_terms = terms
        rank_out = rank

    # Convert to matrix form
    U = [[0] * rank_out for _ in range(n * m)]
    V = [[0] * rank_out for _ in range(m * p)]
    W = [[0] * rank_out for _ in range(p * n)]

    for c, (u_dict, v_dict, w_dict) in enumerate(final_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
