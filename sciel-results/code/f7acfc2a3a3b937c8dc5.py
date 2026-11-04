"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 118  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 6

    # ---- Build a block-structured rank-66 decomposition ----
    # A (3x4) rows split 1+2, cols split 2+2
    # B (4x6) rows split 2+2, cols split 2+2+2
    # Row 0 of C uses 1x2*2x2 (rank 4) blocks; rows 1-2 use Strassen (rank 7)
    # Per column-group: 2*4 (row0) + 2*7 (rows1-2) = 8 + 14 = 22
    # 3 column-groups -> 66 total
    RANK = 66
    U = [[0.0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0.0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0.0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    assert col == RANK, f"expected {RANK}, got {col}"

    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

    # sanity: rank-66 must be valid
    if not verify(U, V, W, RANK):
        # fallback schoolbook (always valid, rank 72)
        rs = n * m * p
        Us = np.zeros((n * m, rs)); Vs = np.zeros((m * p, rs)); Ws = np.zeros((p * n, rs))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, c] = 1
                    Vs[j * p + k, c] = 1
                    Ws[k * n + i, c] = 1
                    c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # ---- Fast column-elimination via tensor-flattening rank test ----
    # M[ijk, l] = U[i m+j, l] * V[j p+k, l] * W[k n+i, l]; the decomposition is
    # valid iff every tensor entry is reproduced, i.e. M @ (col vector) covers ones.
    # A column can be removed iff the remaining columns still span the ones vector.
    def build_M(U_mat, V_mat, W_mat, r):
        M = np.zeros((n * m * p, r))
        for l in range(r):
            colU = U_mat[:, l]
            colV = V_mat[:, l]
            colW = W_mat[:, l]
            for i in range(n):
                base = i * m * p
                for j in range(m):
                    a = colU[i * m + j]
                    if a == 0.0:
                        continue
                    for k in range(p):
                        M[base + j * p + k, l] = a * colV[j * p + k] * colW[k * n + i]
        return M

    def can_reduce(M, target, r, remove_idx):
        # Does the column space of M with column remove_idx dropped still contain target?
        # Equivalent: is target in colspace(M)? removing a column keeps target in
        # colspace only if that column is redundant. Test: rank(M) == rank(M w/o col).
        Mf = np.delete(M, remove_idx, axis=1)
        return np.linalg.matrix_rank(Mf) == np.linalg.matrix_rank(M)

    best_rank = RANK
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    current_rank = RANK
    M = build_M(U, V, W, current_rank)
    target = np.ones(n * m * p)

    for attempt in range(6):
        if time.time() > _deadline - 8:
            break
        if current_rank <= best_rank:
            break
        improved = False
        for remove_idx in range(current_rank):
            if time.time() > _deadline - 5:
                break
            if can_reduce(M, target, current_rank, remove_idx):
                # candidate: column is redundant -> drop and verify exactly
                U_try = np.delete(U, remove_idx, axis=1)
                V_try = np.delete(V, remove_idx, axis=1)
                W_try = np.delete(W, remove_idx, axis=1)
                if verify(U_try, V_try, W_try, current_rank - 1):
                    best_rank = current_rank - 1
                    best_U, best_V, best_W = U_try, V_try, W_try
                    U, V, W = U_try, V_try, W_try
                    current_rank = best_rank
                    M = build_M(U, V, W, current_rank)
                    improved = True
                    break
        if not improved:
            break

    if best_rank < RANK:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
