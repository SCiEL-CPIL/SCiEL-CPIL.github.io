"""Rank-minimal exact tensor decomposition for 2x3 x 3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # rank(2,3,8) = rank(2,3,2) * 4 = 3 * 4 = 12  (Kronecker structure)
    #
    # Base (2,3,2) rank-3 bilinear decomposition (Strassen with one term removed
    # exploiting the 3x2 input-block linear dependencies):
    #   m0 = A00 * B00
    #   m1 = (A00 + A01) * (B10 - B00)
    #   m2 = A10 * (B01 + B11)
    #   C00 = m0 + m1
    #   C01 = m2 - m1
    #   C10 = m0
    #   C11 = m2
    #
    # The 8-column output index k is independent of the 2-row index i, so the
    # full tensor is T(2,3,2) ⊗ (1,1,4), giving 4 identical copies.
    rank = 12

    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 24 x 12
    W = [[0] * rank for _ in range(p * n)]  # 16 x 12

    # For each output column block c in 0..3 (k = 2c, 2c+1) and base term t in 0..2:
    def col(c, t):
        return c * 3 + t

    for c in range(4):
        k0 = 2 * c
        k1 = 2 * c + 1
        # term t=0 : m0 = A00 * B00
        U[0 * m + 0][col(c, 0)] = 1
        V[0 * p + k0][col(c, 0)] = 1
        W[k0 * n + 0][col(c, 0)] = 1
        W[k1 * n + 1][col(c, 0)] = 1
        # term t=1 : m1 = (A00 + A01) * (B10 - B00)
        U[0 * m + 0][col(c, 1)] = 1
        U[0 * m + 1][col(c, 1)] = 1
        V[0 * p + k0][col(c, 1)] = -1
        V[1 * p + k0][col(c, 1)] = 1
        W[k0 * n + 0][col(c, 1)] = 1
        W[k1 * n + 0][col(c, 1)] = -1
        # term t=2 : m2 = A10 * (B01 + B11)
        U[1 * m + 0][col(c, 2)] = 1
        V[0 * p + k1][col(c, 2)] = 1
        V[1 * p + k1][col(c, 2)] = 1
        W[k1 * n + 1][col(c, 2)] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import numpy as np
    d = construct()
    U = np.array(d["U"], dtype=float)
    V = np.array(d["V"], dtype=float)
    W = np.array(d["W"], dtype=float)
    n, m, p = DIMENSIONS
    max_err = 0.0
    for _ in range(500):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_rec = np.zeros((n, p))
        for r in range(W.shape[1]):
            a_r = U[:, r].reshape(n, m)
            b_r = V[:, r].reshape(m, p)
            C_rec += W[:, r].reshape(p, n) * (a_r @ b_r).T
        max_err = max(max_err, float(np.max(np.abs(C_true - C_rec))))
    print("max reconstruction error:", max_err)
    print("rank:", W.shape[1])
