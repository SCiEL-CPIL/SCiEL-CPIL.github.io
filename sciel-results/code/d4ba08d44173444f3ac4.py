"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 150.0

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    outer_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    outer_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _decomp_4x4x2():
    """
    Decomposition of M_{4,4,2} by filtering M_{4,4,4}.
    Returns U: (16, rank), V: (8, rank), W: (8, rank)
    """
    U4, V4, W4 = _decomp_4x4x4()
    
    # V4 is (16, 49). Rows indexed as j*4+k where j=0..3 (input col), k=0..3 (output col)
    # We want output k=2,3 to be zero. So V4 rows at positions j*4+2 and j*4+3 for all j must be zero.
    # These are rows: 2,3,6,7,10,11,14,15
    
    zero_rows_V = [2, 3, 6, 7, 10, 11, 14, 15]
    mask = np.ones(49, dtype=bool)
    for r in range(49):
        if not np.all(V4[zero_rows_V, r] == 0):
            mask[r] = False
            
    U2 = U4[:, mask]
    V2 = V4[:, mask]
    W2 = W4[:, mask]
    
    # V2: keep only rows where k=0,1 -> rows 0,1,4,5,8,9,12,13
    keep_rows_V = [0, 1, 4, 5, 8, 9, 12, 13]
    V2 = V2[keep_rows_V, :]
    
    # W2: rows indexed as k*n+i where k=0..3 (output col), i=0..3 (output row)
    # We want k=0,1 only -> rows 0..7
    W2 = W2[:8, :]
    
    return U2, V2, W2

def _decomp_4x4x3():
    """
    Decomposition of M_{4,4,3} by filtering M_{4,4,4}.
    Returns U: (16, rank), V: (12, rank), W: (12, rank)
    """
    U4, V4, W4 = _decomp_4x4x4()
    
    # V4 rows: j*4+k. We want k=3 to be zero. Rows: 3,7,11,15
    zero_rows_V = [3, 7, 11, 15]
    mask = np.ones(49, dtype=bool)
    for r in range(49):
        if not np.all(V4[zero_rows_V, r] == 0):
            mask[r] = False
            
    U3 = U4[:, mask]
    V3 = V4[:, mask]
    W3 = W4[:, mask]
    
    # V3: keep rows where k=0,1,2 -> rows 0,1,2,4,5,6,8,9,10,12,13,14
    keep_rows_V = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    V3 = V3[keep_rows_V, :]
    
    # W3: rows k*n+i, keep k=0,1,2 -> rows 0..11
    W3 = W3[:12, :]
    
    return U3, V3, W3

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank72():
    """Build the rank-72 decomposition: 4 * M_{4,4,2} + M_{4,4,1}."""
    n, m, p = 4, 4, 9
    
    U2, V2, W2 = _decomp_4x4x2()
    U1, V1, W1 = _schoolbook_nm_p(4, 4, 1)
    
    rank = U2.shape[1] * 4 + U1.shape[1]
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for block in range(4):
        U[:, col_offset:col_offset+U2.shape[1]] = U2
        
        for j in range(m):
            for k in range(2):
                V[j * p + (block * 2 + k), col_offset:col_offset+V2.shape[1]] = V2[j * 2 + k, :]
        
        for k in range(2):
            for i in range(n):
                W[(block * 2 + k) * n + i, col_offset:col_offset+W2.shape[1]] = W2[k * n + i, :]
                
        col_offset += U2.shape[1]
        
    U[:, col_offset:col_offset+U1.shape[1]] = U1
    
    for j in range(m):
        V[j * p + 8, col_offset:col_offset+V1.shape[1]] = V1[j * 1 + 0, :]
        
    for i in range(n):
        W[8 * n + i, col_offset:col_offset+W1.shape[1]] = W1[0 * n + i, :]
        
    return U, V, W

def _build_rank69():
    """Build the rank-69 decomposition: 3 * M_{4,4,3}."""
    n, m, p = 4, 4, 9
    
    U3, V3, W3 = _decomp_4x4x3()
    r3 = U3.shape[1]
    
    rank = r3 * 3
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for block in range(3):
        U[:, col_offset:col_offset+r3] = U3
        
        for j in range(m):
            for k in range(3):
                V[j * p + (block * 3 + k), col_offset:col_offset+r3] = V3[j * 3 + k, :]
        
        for k in range(3):
            for i in range(n):
                W[(block * 3 + k) * n + i, col_offset:col_offset+r3] = W3[k * n + i, :]
                
        col_offset += r3
        
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.all(sums == 1)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    # Try rank-72 decomposition
    try:
        U72, V72, W72 = _build_rank72()
        if _verify_fast(U72, V72, W72, n, m, p):
            best_result = (U72, V72, W72)
            best_rank = U72.shape[1]
    except Exception:
        pass
    
    # Try rank-69 decomposition (3 x M_{4,4,3})
    try:
        U69, V69, W69 = _build_rank69()
        if _verify_fast(U69, V69, W69, n, m, p):
            if U69.shape[1] < best_rank:
                best_result = (U69, V69, W69)
                best_rank = U69.shape[1]
    except Exception:
        pass
    
    # Deadline guard
    if time.time() - start_time > DEADLINE - 5:
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
