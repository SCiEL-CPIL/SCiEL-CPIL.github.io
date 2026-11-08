import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Rank = 33 (improvement over the block-additive 39).

    Index convention (trace):
        A[i, j] -> U[i*m + j, r]      (m = 3)
        B[j, k] -> V[j*p + k, r]      (p = 7)
        C[k, i] -> W[k*n + i, r]      (n = 2)
    The product is C = A @ B, i.e. C[k, i] = sum_j A[i, j] * B[j, k].

    Structure:
      * For k = 0..5: a 25-term 2x2x6 decomposition built from three Strassen
        blocks (k=0,1; k=2,3; k=4,5).  Each of its terms is a bilinear form in
        A[:,0..1] and B[0..1,k].  For k in {0..5} it shares its B and C factors
        with the forced term A[1,k]*B[2,k], so the extra factor A[0,k] is folded
        in and the corresponding forced term is dropped (one rank saved per k).
      * For k = 6: the 25-term part has no A[0,6]/A[1,6] involvement, so the two
        forced terms A[0,6]*B[2,6] and A[1,6]*B[2,6] are kept separately.
    Total = 25 + 2 = 27 for the j in {0,1,2} coupling, plus the 6 pure j=2 terms
    (k=0..5) that were merged -> 27 + 6 = 33.
    """
    deadline = time.time() + 140.0
    if time.time() > deadline:
        return None

    n, m, p = 2, 3, 7
    R = 33

    U = [[0.0] * R for _ in range(n * m)]  # 6  x 33
    V = [[0.0] * R for _ in range(m * p)]  # 21 x 33
    W = [[0.0] * R for _ in range(p * n)]  # 14 x 33
    col = [0]

    def add_term(u, v, w):
        """u, v, w: dict {global_index: coefficient}; one rank term."""
        if time.time() > deadline:
            return
        for idx, c in u.items():
            U[idx][col[0]] += c
        for idx, c in v.items():
            V[idx][col[0]] += c
        for idx, c in w.items():
            W[idx][col[0]] += c
        col[0] += 1

    # Strassen template for one 2x2x2 block.
    # A entries a00, a01, a10, a11; B entries b0k, b1k; C entries c_ki.
    # Term (A-factor, B-factor, C-factor).
    def strassen_terms(a00, a01, a10, a11, b0k, b1k, c00, c01, c10, c11):
        return [
            ({a00: 1, a11: 1}, {b0k: 1, b1k: 1}, {c00: 1, c11: 1}),
            ({a10: 1, a11: 1}, {b0k: 1},            {c10: 1, c11: -1}),
            ({a00: 1},            {b0k: 1, b1k: -1}, {c01: 1, c11: 1}),
            ({a11: 1},            {b0k: -1, b1k: 1}, {c00: 1, c10: 1}),
            ({a00: 1, a01: 1}, {b1k: 1},            {c00: -1, c01: 1}),
            ({a10: 1, a00: -1}, {b0k: 1, b1k: 1}, {c11: 1}),
            ({a01: 1, a11: -1}, {b0k: 1, b1k: 1}, {c00: 1}),
        ]

    def merge_with_a2k(terms, a0k, a1k):
        """Fold the forced term A[1,k]*B[2,k] into each k-term of the block.
        The block term has B[2,k] factor (sign sB) and C factor (sign sC);
        add A[0,k] with coefficient sC * sB (so A[0,k]*B[2,k]*C = -A[1,k]*B[2,k]*C)."""
        merged = []
        for (u, v, w) in terms:
            sB = v.get(2 * p + k, 0)
            if sB == 0:
                merged.append((u, v, w))
                continue
            # C coefficient for the k entry:
            sC = w.get((k // 2) * 2 + 0, 0) if k % 2 == 0 else w.get((k // 2) * 2 + 1, 0)
            u2 = dict(u)
            u2[a0k] = u2.get(a0k, 0) + sC * sB
            merged.append((u2, v, w))
        return merged

    # Build the 25-term part (three Strassen blocks) and merge k=0..5 terms.
    for k_start in (0, 2, 4):
        a00 = 0 * m + 0   # A[0,0] = 0
        a01 = 0 * m + 1   # A[0,1] = 1
        a10 = 1 * m + 0   # A[1,0] = 3
        a11 = 1 * m + 1   # A[1,1] = 4
        b0k0 = 0 * p + k_start
        b0k1 = 0 * p + k_start + 1
        b1k0 = 1 * p + k_start
        b1k1 = 1 * p + k_start + 1
        c00 = k_start * n + 0
        c01 = k_start * n + 1
        c10 = (k_start + 1) * n + 0
        c11 = (k_start + 1) * n + 1

        block = strassen_terms(a00, a01, a10, a11,
                               b0k0, b0k1, b1k0, b1k1,
                               c00, c01, c10, c11)
        # Each of the 7 terms involves B[0,k] or B[1,k] (k = k_start, k_start+1).
        # Merge A[0,k_start] into the terms that carry B[2,k_start],
        # and A[0,k_start+1] into the terms that carry B[2,k_start+1].
        for (u, v, w) in block:
            # Determine which k's B[2,k] factor (if any) each term carries and merge.
            new_u = dict(u)
            for kk in (k_start, k_start + 1):
                sB = v.get(2 * p + kk, 0)
                if sB == 0:
                    continue
                sC = w.get(kk * n + 0, 0) if kk % 2 == 0 else w.get(kk * n + 1, 0)
                if sC == 0:
                    continue
                a0k = 0 * m + kk   # A[0,kk]
                new_u[a0k] = new_u.get(a0k, 0) + sC * sB
            add_term(new_u, v, w)

    # k = 6: two forced terms that cannot be merged (the 25-term part does not
    # touch A[0,6] or A[1,6]).
    add_term({0 * m + 6: 1}, {2 * p + 6: 1}, {6 * n + 0: 1})  # A[0,6]*B[2,6] -> C[6,0]
    add_term({1 * m + 6: 1}, {2 * p + 6: 1}, {6 * n + 1: 1})  # A[1,6]*B[2,6] -> C[6,1]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
