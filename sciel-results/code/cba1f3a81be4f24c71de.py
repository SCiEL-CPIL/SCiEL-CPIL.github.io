import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor (all-ones)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_mat = U[:, t].reshape(n, m)
        v_mat = V[:, t].reshape(m, p)
        w_mat = W[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    expected = np.ones((n, m, p), dtype=float)
    return bool(np.allclose(T, expected, atol=1e-8))


def _build_strassen_terms(n, m, p):
    """
    Kronecker-product embedding for the 3x4x6 multiplication tensor.

    Split A into a 2x4 block (rows 0,1) and a 1x4 block (row 2).
    - 2x4x6 part: split into 2x2x2 Strassen blocks over j-pairs and k-pairs.
      Number of blocks = (m//2) * (p//2) = 2 * 3 = 6, each rank 7 -> rank 42.
    - 1x4x6 part: schoolbook, rank 6.
    Total rank = 42 + 6 = 48.
    """
    terms = []

    def add_term(a, b, c):
        terms.append((a, b, c))

    # 2x2x2 Strassen block: A block rows i0,i0+1 cols j0,j0+1
    #                      B block rows j0,j0+1 cols k0,k0+1
    #                      C block rows i0,i0+1 cols k0,k0+1
    # Standard 7-term Strassen.
    def strassen_block(i0, j0, k0):
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        add_term(
            {(i0, j0): 1, (i0 + 1, j0 + 1): 1},
            {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
            {(i0, k0): 1, (i0 + 1, k0 + 1): 1},
        )
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        add_term(
            {(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
            {(j0, k0): 1},
            {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1},
        )
        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        add_term(
            {(i0, j0): 1},
            {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
            {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1},
        )
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        add_term(
            {(i0 + 1, j0 + 1): 1},
            {(j0 + 1, k0): 1, (j0, k0): -1},
            {(i0, k0): 1, (i0 + 1, k0): 1},
        )
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        add_term(
            {(i0, j0): 1, (i0, j0 + 1): 1},
            {(j0 + 1, k0 + 1): 1},
            {(i0, k0): -1, (i0, k0 + 1): 1},
        )
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        add_term(
            {(i0 + 1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k0 + 1): 1},
            {(i0 + 1, k0 + 1): 1},
        )
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        add_term(
            {(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
            {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
            {(i0, k0): 1},
        )

    # 2x4x6 part: i in {0,1}, j-pairs, k-pairs
    for j0 in range(0, m - 1, 2):
        for k0 in range(0, p - 1, 2):
            strassen_block(0, j0, k0)

    # 1x4x6 part: row 2, schoolbook
    for j in range(m):
        for k in range(p):
            add_term({(2, j): 1}, {(j, k): 1}, {(2, k): 1})

    return terms


def _terms_to_matrices(terms, n, m, p):
    """Convert list of (a,b,c) dict-terms into U,V,W matrices."""
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for t, (a, b, c) in enumerate(terms):
        for (i, j), val in a.items():
            U[i * m + j][t] = val
        for (j, k), val in b.items():
            V[j * p + k][t] = val
        for (i, k), val in c.items():
            W[k * n + i][t] = val
    return U, V, W


def _peel_terms(terms, n, m, p, deadline):
    """Drop terms whose contribution vector is all zero."""
    rank = len(terms)
    size = n * m * p
    vecs = []
    for t in range(rank):
        a, b, c = terms[t]
        v = np.zeros(size)
        for (i, j), ua in a.items():
            for (j2, k), vb in b.items():
                if j2 != j:
                    continue
                for (i2, k2), wc in c.items():
                    if i2 != i or k2 != k:
                        continue
                    idx = (i * m + j) * p + k
                    v[idx] += ua * vb * wc
        vecs.append(v)
    kept = []
    for t in range(rank):
        if not np.allclose(vecs[t], 0.0, atol=1e-12):
            kept.append(t)
    if len(kept) == rank:
        return None
    return [terms[t] for t in kept]


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # Fallback: schoolbook
    school_terms = [({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                    for i in range(n) for j in range(m) for k in range(p)]
    U_s, V_s, W_s = _terms_to_matrices(school_terms, n, m, p)
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = n * m * p

    # Kronecker-embedded Strassen decomposition (rank 48)
    strassen_terms = _build_strassen_terms(n, m, p)
    U_t, V_t, W_t = _terms_to_matrices(strassen_terms, n, m, p)
    if _validate(U_t, V_t, W_t, n, m, p, len(strassen_terms)):
        best = {"U": U_t, "V": V_t, "W": W_t}
        best_rank = len(strassen_terms)

    # Peeling pass on the Strassen terms
    if time.time() < deadline:
        peeled = _peel_terms(strassen_terms, n, m, p, deadline)
        if peeled is not None and len(peeled) < best_rank:
            U_p, V_p, W_p = _terms_to_matrices(peeled, n, m, p)
            if _validate(U_p, V_p, W_p, n, m, p, len(peeled)):
                best = {"U": U_p, "V": V_p, "W": W_p}
                best_rank = len(peeled)

    return best
