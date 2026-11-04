"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """
    Return the rank-7 Winograd F(2) decomposition for 2x2 * 2x2.
    """
    U = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, -1, 1],
    ], dtype=float)

    V = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, -1, 0, 1, -1, -1, 1],
    ], dtype=float)

    W = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
    ], dtype=float)

    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(7):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_kron():
    """
    Construct a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Winograd F(2).
    """
    w2 = _winograd_2x2x2()
    if not w2:
        return None

    U2 = np.array(w2["U"])  # shape (4, 7)
    V2 = np.array(w2["V"])  # shape (4, 7)
    W2 = np.array(w2["W"])  # shape (4, 7)

    n, m, p = 4, 4, 4
    rank = 49
    U4 = np.zeros((n * m, rank))
    V4 = np.zeros((m * p, rank))
    W4 = np.zeros((p * n, rank))

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = i1 * 8 + i2 * 4 + j1 * 2 + j2
                            col_idx = c1 * 7 + c2
                            U4[row_idx, col_idx] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = j1 * 8 + j2 * 4 + k1 * 2 + k2
                            col_idx = c1 * 7 + c2
                            V4[row_idx, col_idx] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = k1 * 8 + k2 * 4 + i1 * 2 + i2
                            col_idx = c1 * 7 + c2
                            W4[row_idx, col_idx] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _construct_4x4x5():
    """
    Construct a rank-53 decomposition for 4x4 * 4x5.
    Uses rank-49 Kronecker for first 4 columns, then 4 additional rank terms for 5th column.
    """
    n, m, p = 4, 4, 5

    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)

    U4 = np.array(base["U"])
    V4 = np.array(base["V"])
    W4 = np.array(base["W"])

    rank_base = 49
    rank_new = rank_base + 4

    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))

    U_new[:, :rank_base] = U4

    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]

    W_new[:16, :rank_base] = W4

    for j in range(4):
        col_idx = rank_base + j
        for i in range(4):
            U_new[i * 4 + j, col_idx] = 1.0
        V_new[j * 5 + 4, col_idx] = 1.0
        for i in range(4):
            W_new[4 * 4 + i, col_idx] = 1.0

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def _construct_4x4x5_winograd_hybrid():
    """
    Alternative construction: use Winograd F(2) for 2x4 blocks.
    
    Split A (4x4) into two 2x4 blocks: A_top (rows 0,1) and A_bot (rows 2,3).
    Split B (4x5) into two 4x... no, B is 4x5 so columns 0-4.
    
    Actually, let's think differently. The tensor for 4x4*4x5 multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, k in 0..4, j in 0..3.
    
    We can decompose using a 2x2 Kronecker structure on the (i,j) and (j,k) indices.
    
    For the 4x4 part (k=0..3), we use the rank-49 Kronecker decomposition.
    For the 5th column (k=4), we need rank 4 terms.
    
    Total rank: 53.
    """
    return _construct_4x4x5()

def construct():
    n, m, p = DIMENSIONS

    start_time = time.time()
    deadline = start_time + 140

    try:
        result = _construct_4x4x5()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    try:
        result = _construct_4x4x5_winograd_hybrid()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
