# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 130

def _validate_decomposition(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=1e-10)

def _build_2x2x3_rank11():
    """Build a rank-11 decomposition for 2x2x3."""
    rank = 11
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Strassen for cols 0,1 (7 terms)
    strassen = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3, 1, 2], [1, 1, -1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, -1]),
        ([0, 1], [1, 1], [3], [1], [1, 3], [1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [0, 2], [-1, 1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 1], [1, 1]),
    ]
    
    for t, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(strassen):
        for ui, uc in zip(u_idx, u_c):
            U[ui, t] += uc
        for vi, vc in zip(v_idx, v_c):
            V[vi, t] += vc
        for wi, wc in zip(w_idx, w_c):
            W[wi, t] += wc
    
    # Schoolbook for col 2 (4 terms)
    # V[0*3+2]=V[2], V[1*3+2]=V[5]
    # W[2*2+0]=W[4], W[2*2+1]=W[5]
    schoolbook = [
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([2], [1], [2], [1], [5], [1]),
        ([3], [1], [5], [1], [5], [1]),
    ]
    
    for t, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(schoolbook):
        col = 7 + t
        for ui, uc in zip(u_idx, u_c):
            U[ui, col] += uc
        for vi, vc in zip(v_idx, v_c):
            V[vi, col] += vc
        for wi, wc in zip(w_idx, w_c):
            W[wi, col] += wc
    
    # Validate
    T = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T[i*2+j, j*3+k, k*2+i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(T, T_recon, atol=1e-10), "2x2x3 rank-11 decomposition failed"
    
    return U, V, W

def _build_2x2x4_rank14():
    """Build a rank-14 decomposition for 2x2x4 (two Strassen blocks)."""
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    strassen = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3, 1, 2], [1, 1, -1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, -1]),
        ([0, 1], [1, 1], [3], [1], [1, 3], [1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [0, 2], [-1, 1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 1], [1, 1]),
    ]
    
    for block in range(2):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_idx, u_c, v_idx, v_c, w_idx, w_c = strassen[t]
            for ui, uc in zip(u_idx, u_c):
                U[ui, col] += uc
            for vi_off, vc in zip(v_idx, v_c):
                V[2 * block + vi_off, col] += vc
            for wi_off, wc in zip(w_idx, w_c):
                W[2 * block + wi_off, col] += wc
    
    # Validate
    T = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                T[i*2+j, j*4+k, k*2+i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(T, T_recon, atol=1e-10), "2x2x4 rank-14 decomposition failed"
    
    return U, V, W

def _build_full_decomposition():
    """
    Build the full 2x2x9 decomposition.
    Strategy: 2x2x4 (rank 14) + 2x2x5 (rank 18) = 32.
    Or try: 3x(2x2x3) = 3*11 = 33.
    Let's go with 2x2x4 + 2x2x5 = 32 for now, and try to optimize.
    """
    n, m, p = 2, 2, 9
    
    # Use 2x2x4 for cols 0-3 and 2x2x5 for cols 4-8
    U1, V1, W1 = _build_2x2x4_rank14()
    U2, V2, W2 = _build_2x2x5_rank18()
    
    # Combine
    rank = 14 + 18
    U = np.zeros((4, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Copy block 1 (cols 0-3)
    U[:, :14] = U1
    # V1 has shape (8, 14), V has shape (18, rank)
    # V1[j*4+k, r] maps to V[j*9+k, r] for k=0..3
    for j in range(2):
        for k in range(4):
            V[j*9+k, :14] = V1[j*4+k, :]
    # W1 has shape (8, 14), W has shape (18, rank)
    # W1[k*2+i, r] maps to W[k*2+i, r] for k=0..3
    for k in range(4):
        for i in range(2):
            W[k*2+i, :14] = W1[k*2+i, :]
    
    # Copy block 2 (cols 4-8)
    U[:, 14:32] = U2
    for j in range(2):
        for k in range(5):
            V[j*9+(4+k), 14:32] = V2[j*5+k, :]
    for k in range(5):
        for i in range(2):
            W[(4+k)*2+i, 14:32] = W2[k*2+i, :]
    
    # Validate
    T = np.zeros((4, 18, 18), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(9):
                T[i*2+j, j*9+k, k*2+i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, T_recon, atol=1e-10):
        return None, None, None
    
    return U, V, W

def _build_2x2x5_rank18():
    """Build a rank-18 decomposition for 2x2x5."""
    rank = 18
    U = np.zeros((4, rank))
    V = np.zeros((10, rank))
    W = np.zeros((10, rank))
    
    # Use 2x2x3 for cols 0-2 (rank 11) + 2x2x2 for cols 3-4 (rank 7)
    U3, V3, W3 = _build_2x2x3_rank11()
    U2, V2, W2 = _build_2x2x2_rank7()
    
    U[:, :11] = U3
    U[:, 11:18] = U2
    
    for j in range(2):
        for k in range(3):
            V[j*5+k, :11] = V3[j*3+k, :]
        for k in range(2):
            V[j*5+(3+k), 11:18] = V2[j*2+k, :]
    
    for k in range(3):
        for i in range(2):
            W[k*2+i, :11] = W3[k*2+i, :]
    for k in range(2):
        for i in range(2):
            W[(3+k)*2+i, 11:18] = W2[k*2+i, :]
    
    # Validate
    T = np.zeros((4, 10, 10), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(5):
                T[i*2+j, j*5+k, k*2+i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(T, T_recon, atol=1e-10), "2x2x5 rank-18 decomposition failed"
    
    return U, V, W

def _build_2x2x2_rank7():
    """Build a rank-7 decomposition for 2x2x2 (Strassen)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    strassen = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3, 1, 2], [1, 1, -1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, -1]),
        ([0, 1], [1, 1], [3], [1], [1, 3], [1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [0, 2], [-1, 1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 1], [1, 1]),
    ]
    
    for t, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(strassen):
        for ui, uc in zip(u_idx, u_c):
            U[ui, t] += uc
        for vi, vc in zip(v_idx, v_c):
            V[vi, t] += vc
        for wi, wc in zip(w_idx, w_c):
            W[wi, t] += wc
    
    # Validate
    T = np.zeros((4, 4, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1.0
    
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(T, T_recon, atol=1e-10), "2x2x2 rank-7 decomposition failed"
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    remaining = _TIMEOUT - elapsed
    
    # Try the block decomposition
    if remaining > 5:
        U, V, W = _build_full_decomposition()
        if U is not None:
            rank = U.shape[1]
            U_list = [[U[i][j] for j in range(rank)] for i in range(n * m)]
            V_list = [[V[i][j] for j in range(rank)] for i in range(m * p)]
            W_list = [[W[i][j] for j in range(rank)] for i in range(p * n)]
            if _validate_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
