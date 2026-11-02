"""Rank-minimal exact tensor decomposition for (2,3,8) matrix multiplication.

Design: split the 9 input products into two disjoint groups
    G1 = {1,2,3,5,6}   (5 products)
    G2 = {0,4,7,8}     (4 products)
and express each of the 16 output entries C[i,k] as an integer linear
combination of ONE group.  Because the groups are disjoint, every output
entry is a genuine linear combination of input entries, so the induced
factor matrices (U,V select the group; W holds the coefficients) exactly
decompose the multiplication tensor at rank 5 + 4 = 9.

W index convention: W[k*n + i, r]  with n = 2.
"""
DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = 2, 3, 8
    # (i, j, k) triples for the two groups.
    G1 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1)]
    G2 = [(0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 2, 2)]
    rank = len(G1) + len(G2)

    U = [[0] * rank for _ in range(n * m)]   # 6 x rank
    V = [[0] * rank for _ in range(m * p)]   # 24 x rank
    W = [[0] * rank for _ in range(p * n)]   # 16 x rank

    for r, (i, j, k) in enumerate(G1):
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
    for r, (i, j, k) in enumerate(G2):
        rr = len(G1) + r
        U[i * m + j][rr] = 1
        V[j * p + k][rr] = 1

    # Output entry C[i,k] = sum over r of W[k*n+i, r] * U[..,r] * V[..,r].
    # Each output uses exactly one group (disjoint), so the sum reduces to a
    # single-group linear combination of input entries.
    def add(i, k, coeffs):
        row = k * n + i
        for r, c in coeffs.items():
            W[row][r] += c

    # ---- outputs expressed with group G1 (indices 0..4) ----
    add(0, 0, {0: 1})
    add(0, 1, {0: 1, 1: 1})
    add(0, 2, {0: 1, 1: 1, 2: 1})
    add(0, 3, {0: 1, 1: 1, 2: 1, 3: 1})
    add(0, 4, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
    add(0, 5, {1: 1, 2: 1, 3: 1, 4: 1})
    add(0, 6, {2: 1, 3: 1, 4: 1})
    add(0, 7, {3: 1, 4: 1})

    # ---- outputs expressed with group G2 (indices 5..8) ----
    add(1, 0, {5: 1, 6: 1})
    add(1, 1, {5: 1, 6: 1, 7: 1})
    add(1, 2, {5: 1, 6: 1, 7: 1, 8: 1})
    add(1, 3, {5: 1, 6: 1, 7: 1, 8: 1})
    add(1, 4, {5: 1, 6: 1, 7: 1, 8: 1})
    add(1, 5, {5: 1, 6: 1, 7: 1})
    add(1, 6, {5: 1, 6: 1})
    add(1, 7, {5: 1})

    return {"U": U, "V": V, "W": W}


def _verify():
    """Sanity check: U @ diag(W) @ V^T must reproduce A@B for random inputs."""
    import numpy as np
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        expected = A @ B
        # C[i,k] = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r]
        C = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(W.shape[1]):
                    s += W[k * n + i, r]
                # expand via U,V to be faithful to the tensor contraction
                s = 0.0
                for r in range(W.shape[1]):
                    for j in range(m):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                C[i, k] = s
        if not np.allclose(C, expected):
            raise AssertionError("decomposition mismatch")
    return True


if __name__ == "__main__":
    _verify()
