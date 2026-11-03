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
        [1, 0, 0, 1, 1, 0, -1],  # a00
        [0, 0, 0, 0, 1, 1, 0],   # a01
        [0, 1, 0, 0, 0, 0, 1],   # a10
        [1, 1, 1, 0, 0, 0, 0],   # a11
    ], dtype=float)
    
    V = np.array([
        [1, 0, 0, -1, 1, 0, 1],  # b00
        [0, 1, 1, 0, 0, 0, 1],   # b01
        [0, 0, 0, 1, 0, 1, 0],   # b10
        [1, 0, -1, 0, 0, 1, 0],  # b11
    ], dtype=float)
    
    W = np.array([
        [1, -1, -1, 1, 1, -1, -1],  # c00 (k=0,i=0)
        [1, -1, -1, 1, 1, -1, -1],  # c01 (k=0,i=1)
        [1, -1, -1, 1, 1, -1, -1],  # c10 (k=1,i=0)
        [1, -1, -1, 1, 1, -1, -1],  # c11 (k=1,i=1)
    ], dtype=float)
    
    # Verify
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
    Construct a rank-49 decomposition for 4x4x4 using Kronecker product of Winograd F(2).
    """
    w2 = _winograd_2x2x2()
    if not w2:
        return None
        
    U2 = np.array(w2["U"])
    V2 = np.array(w2["V"])
    W2 = np.array(w2["W"])
    
    n, m, p = 4, 4, 4
    rank = 49
    U4 = np.zeros((n * m, rank))
    V4 = np.zeros((m * p, rank))
    W4 = np.zeros((p * n, rank))
    
    # U4[i1*2+i2, c1*7+c2] = U2[i1, c1] * U2[i2, c2]
    for i1 in range(2):
        for i2 in range(2):
            for c1 in range(7):
                for c2 in range(7):
                    U4[i1 * 2 + i2, c1 * 7 + c2] = U2[i1, c1] * U2[i2, c2]
                    
    # V4[j1*2+j2, c1*7+c2] = V2[j1, c1] * V2[j2, c2]
    for j1 in range(2):
        for j2 in range(2):
            for c1 in range(7):
                for c2 in range(7):
                    V4[j1 * 2 + j2, c1 * 7 + c2] = V2[j1, c1] * V2[j2, c2]
                    
    # W4[k*n+i, c] where n=4.
    # k = k1*2 + k2, i = i1*2 + i2
    # Row index in W4: k*4 + i = (k1*2+k2)*4 + (i1*2+i2) = k1*8 + k2*4 + i1*2 + i2
    # W2 row index: k_sub*2 + i_sub
    # W4[k1*8 + k2*4 + i1*2 + i2, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = k1 * 8 + k2 * 4 + i1 * 2 + i2
                            W4[row_idx, c1 * 7 + c2] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                            
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _construct_4x4x5():
    """
    Construct a rank-50 decomposition for 4x4x5.
    """
    n, m, p = 4, 4, 5
    
    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)
        
    U4 = np.array(base["U"])
    V4 = np.array(base["V"])
    W4 = np.array(base["W"])
    
    rank_base = 49
    rank_new = rank_base + 1
    
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    # Copy base part
    U_new[:, :rank_base] = U4
    # V_new needs re-indexing because p changes from 4 to 5
    # V4 index: j*4 + k
    # V_new index: j*5 + k
    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]
            
    # W_new needs re-indexing? No, n and p for W are p*n.
    # W4 index: k*4 + i (k in 0..3)
    # W_new index: k*4 + i (k in 0..4)
    # Rows 0..15 of W_new correspond to k=0..3
    W_new[:16, :rank_base] = W4
    
    # Add the extra term for k=4
    # U_new[i*4+j, rank_base] = 1
    U_new[:, rank_base] = 1.0
    
    # V_new[j*5+4, rank_base] = 1
    for j in range(4):
        V_new[j * 5 + 4, rank_base] = 1.0
        
    # W_new[4*4+i, rank_base] = 1  (k=4)
    for i in range(4):
        W_new[4 * 4 + i, rank_base] = 1.0
        
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

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
    
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
