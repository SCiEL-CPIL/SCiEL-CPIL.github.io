import time
import numpy as np

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 100

    n, m, p = 2, 2, 12

    # Fallback schoolbook rank-48
    fb = _schoolbook(n, m, p)

    # Classic Strassen rank-7 decomposition for 2x2x2 matrix multiplication
    # U rows: A00, A01, A10, A11 (index i*m+j)
    # V rows: B00, B01, B10, B11 (index j*p+k)
    # W rows: C00, C01, C10, C11 (index k*n+i)
    U_s = np.array([
        [ 1,  1,  0,  0,  1,  0, -1],
        [ 0,  1,  1,  0,  0,  1,  0],
        [ 0,  0,  1,  1,  0,  0,  1],
        [ 0,  0,  1,  1,  1, -1,  0]
    ], dtype=float)

    V_s = np.array([
        [ 1,  0,  0,  1,  1,  1,  1],
        [ 0,  0,  1,  0,  0,  0,  1],
        [ 0,  1,  0,  0,  0,  1,  0],
        [ 1,  0,  0, -1,  1,  1,  0]
    ], dtype=float)

    W_s = np.array([
        [ 1,  0,  0,  0, -1, -1,  1],
        [ 0,  1,  1,  0,  0,  1,  0],
        [ 0,  0,  0,  1,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  1,  1]
    ], dtype=float)

    # Verify the Strassen block for 2x2x2
    def _verify_block(U_b, V_b, W_b, ns, ms, ps):
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    s = 0.0
                    for r in range(len(U_b[0])):
                        s += U_b[i*ms+j][r] * V_b[j*ps+k][r] * W_b[k*ns+i][r]
                    if abs(s - (1.0 if j == i else 0.0)) > 1e-10:
                        return False
        return True

    if not _verify_block(U_s.tolist(), V_s.tolist(), W_s.tolist(), 2, 2, 2):
        return fb

    # Build rank-42 decomposition: 6 groups of 7 (one Strassen block per 2-column slice of B)
    num_groups = p // 2  # 6 groups
    rank = num_groups * 7  # 42

    U_full = np.zeros((n * m, rank), dtype=float)
    V_full = np.zeros((m * p, rank), dtype=float)
    W_full = np.zeros((p * n, rank), dtype=float)

    for g in range(num_groups):
        for r in range(7):
            col = g * 7 + r
            # U: A factors, identical across all groups
            U_full[:, col] = U_s[:, r]

            # V: B factors for this group (columns 2g, 2g+1 of B)
            for j in range(m):
                for k in range(2):
                    v_row = j * p + (2 * g + k)
                    V_full[v_row, col] = V_s[j * 2 + k, r]

            # W: C factors for this group (rows 2g, 2g+1 of C)
            for k in range(2):
                for i in range(n):
                    w_row = (2 * g + k) * n + i
                    W_full[w_row, col] = W_s[k * 2 + i, r]

    # Verify full decomposition
    if _check(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        if time.time() < deadline:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    return fb
# EVOLVE-BLOCK-END
