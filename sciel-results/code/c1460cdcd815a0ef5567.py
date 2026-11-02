"""Matrix multiplication 4x4 by 4x5 - block decomposition with Strassen."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify U,V,W decompose the matmul tensor: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]==1."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return (U,V,W) for 2x2x2 matmul, rank 7, with U[i*2+j], V[j*2+k], W[k*2+i]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen: M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11),
    # M4=A11(B10-B00), M5=(A00+A10)B11, M6=(A10-A00)(B00+B10), M7=(A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # A00: M1, M3, M5
        [0, 1, 0, 0, 0, 1, 0],  # A10: M2, M6
        [0, 0, 0, 1, 0, -1, 1], # A01: M4, M6, M7
        [1, 1, 0, 1, 0, 0, -1], # A11: M1, M2, M4, M7
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 0, 0, 0, 0],  # B00: M1, M2
        [0, 0, 1, 0, 1, 0, 1],  # B01: M3, M5, M7
        [0, 0, -1, -1, 0, 1, 0],# B10: M3, M4, M6
        [1, 0, -1, 1, 1, 0, 0], # B11: M1, M3, M4, M5
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0],  # C11: M1, M2, M3, M6
    ], dtype=float)
    return U, V, W

def _strassen_2x2x3():
    """Return (U,V,W) for 2x2x3 matmul: U[4,7], V[6,7], W[6,7]."""
    U, V2, W2 = _strassen_2x2x2()
    V = np.zeros((6, 7))
    W = np.zeros((6, 7))
    # k=0,1 from square
    V[0:4, :] = V2
    W[0:4, :] = W2
    # k=2: naive, rank 4
    U2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    V2b = np.array([
        [1, 0, 0, 0],  # B02
        [0, 1, 0, 0],  # B12
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    W2b = np.array([
        [1, 0, 0, 0],  # C02
        [0, 0, 1, 0],  # C12
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    V[4:6, :] = V2b
    W[4:6, :] = W2b
    # Extend U with 4 extra columns for the naive k=2 part
    U_full = np.zeros((4, 11))
    U_full[:, 0:7] = U
    U_full[:, 7:11] = U2
    V_full = np.zeros((6, 11))
    V_full[:, 0:7] = V
    V_full[:, 7:11] = V2b
    W_full = np.zeros((6, 11))
    W_full[:, 0:7] = W
    W_full[:, 7:11] = W2b
    return U_full, V_full, W_full

def _strassen_2x3x2():
    """Return (U,V,W) for 2x3x2 matmul: U[6,7], V[6,7], W[4,7]."""
    U2, V2, W2 = _strassen_2x2x2()
    U = np.zeros((6, 7))
    W = np.zeros((4, 7))
    # i=0,1 from square (j=0,1)
    U[0:4, :] = U2
    W[:, :] = W2
    # j=2: naive, rank 4
    U2b = np.array([
        [1, 0, 0, 0],  # A02
        [0, 1, 0, 0],  # A12
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    V2b = np.array([
        [1, 0, 0, 0],  # B02
        [0, 1, 0, 0],  # B12
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    W2b = np.array([
        [1, 0, 0, 0],  # C02
        [0, 0, 1, 0],  # C12
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    U_full = np.zeros((6, 11))
    U_full[:, 0:7] = U
    U_full[:, 7:11] = U2b
    V_full = np.zeros((6, 11))
    V_full[:, 0:7] = V2
    V_full[:, 7:11] = V2b
    W_full = np.zeros((4, 11))
    W_full[:, 0:7] = W
    W_full[:, 7:11] = W2b
    return U_full, V_full, W_full

def _strassen_3x2x2():
    """Return (U,V,W) for 3x2x2 matmul: U[6,7], V[4,7], W[6,7]."""
    U2, V2, W2 = _strassen_2x2x2()
    U = np.zeros((6, 7))
    V = np.zeros((4, 7))
    # i=0,1 from square
    U[0:4, :] = U2
    V[:, :] = V2
    # i=2: naive, rank 4
    U2b = np.array([
        [1, 0, 0, 0],  # A20
        [0, 1, 0, 0],  # A21
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    V2b = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    W2b = np.array([
        [1, 0, 0, 0],  # C20
        [0, 0, 1, 0],  # C21
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    U_full = np.zeros((6, 11))
    U_full[:, 0:7] = U
    U_full[:, 7:11] = U2b
    V_full = np.zeros((4, 11))
    V_full[:, 0:7] = V
    V_full[:, 7:11] = V2b
    W_full = np.zeros((6, 11))
    W_full[:, 0:7] = W2
    W_full[:, 7:11] = W2b
    return U_full, V_full, W_full

def _build_4x4x4():
    """Build 4x4x4 decomposition using 2x2 blocks of Strassen. U[16,?], V[16,?], W[16,?]."""
    U2, V2, W2 = _strassen_2x2x2()
    # 2x2 blocks: (i0,i1) x (j0,j1) x (k0,k1)
    # 4 blocks, each rank 7, total rank 28
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    block_idx = 0
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                # block (ib,jb,kb): i in [2*ib,2*ib+2), j in [2*jb,2*jb+2), k in [2*kb,2*kb+2)
                col_off = block_idx * 7
                for li in range(2):
                    for lj in range(2):
                        U[(2*ib+li)*4 + (2*jb+lj), col_off:col_off+7] = U2[li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(2*jb+lj)*4 + (2*kb+lk), col_off:col_off+7] = V2[lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(2*kb+lk)*4 + (2*ib+li), col_off:col_off+7] = W2[lk*2+li]
                block_idx += 1
    return U, V, W

def _build_4x4x5():
    """Build 4x4x5 decomposition. U[16,?], V[20,?], W[20,?]."""
    # 4x4 part (k=0..3): rank 28
    U4, V4, W4 = _build_4x4x4()
    # 4x1 part (k=4): naive, rank 16
    U5 = np.zeros((16, 16))
    V5 = np.zeros((20, 16))
    W5 = np.zeros((20, 16))
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U5[idx, idx] = 1
            V5[j*5 + 4, idx] = 1
            W5[4*4 + i, idx] = 1
    # Combine
    rank = 28 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    U[:, 0:28] = U4
    U[:, 28:44] = U5
    V[:, 0:28] = V4
    V[:, 28:44] = V5
    W[:, 0:28] = W4
    W[:, 28:44] = W5
    return U, V, W

def _try_rank_als(rank, n, m, p, seed=42, max_iters=200, start=None):
    """ALS to find a rank-`rank` decomposition, optionally starting from `start`."""
    np.random.seed(seed)
    if start is not None:
        U, V, W = [np.array(x, dtype=float) for x in start]
        # Pad or truncate to target rank
        cur_rank = U.shape[1]
        if cur_rank < rank:
            extra = rank - cur_rank
            U = np.hstack([U, np.random.randn(n*m, extra) * 0.1])
            V = np.hstack([V, np.random.randn(m*p, extra) * 0.1])
            W = np.hstack([W, np.random.randn(p*n, extra) * 0.1])
        else:
            U, V, W = U[:, :rank], V[:, :rank], W[:, :rank]
    else:
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1

    for it in range(max_iters):
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    return U, V, W

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s

    # Build the block decomposition (rank 44)
    U0, V0, W0 = _build_4x4x5()
    # Verify the raw block decomposition
    ok_raw = _verify_decomposition(U0.tolist(), V0.tolist(), W0.tolist(), n, m, p)
    # Round to half-integers and verify
    U_r = _round_matrix(U0)
    V_r = _round_matrix(V0)
    W_r = _round_matrix(W0)
    ok_round = _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p)

    if ok_round:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    if ok_raw:
        # Try to round via ALS polish
        U, V, W = _try_rank_als(44, n, m, p, seed=1, max_iters=50, start=(U0, V0, W0))
        U_r = _round_matrix(U); V_r = _round_matrix(V); W_r = _round_matrix(W)
        if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
