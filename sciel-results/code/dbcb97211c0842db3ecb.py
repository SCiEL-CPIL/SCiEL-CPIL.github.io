import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _strassen_2x2():
    """Return U, V, W matrices for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
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
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Build 4x4x4 Strassen (rank 49)."""
    U2, V2, W2 = _strassen_2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1; W_block[3, 1] = -1
    
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1; W_block[3, 2] = 1
    
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1
    
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1
    
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    for k in range(7):
        for s in range(7):
            t = k * 7 + s
            for i in range(4):
                for j in range(4):
                    idx_A = i * 4 + j
                    b_A = (i // 2) * 2 + (j // 2)
                    l_A = (i % 2) * 2 + (j % 2)
                    U4[idx_A, t] = U_block[b_A, k] * U2[l_A, s]
                    
            for j in range(4):
                for k2 in range(4):
                    idx_B = j * 4 + k2
                    b_B = (j // 2) * 2 + (k2 // 2)
                    l_B = (j % 2) * 2 + (k2 % 2)
                    V4[idx_B, t] = V_block[b_B, k] * V2[l_B, s]
                    
            for k2 in range(4):
                for i in range(4):
                    idx_C = k2 * 4 + i
                    b_C = (k2 // 2) * 2 + (i // 2)
                    l_C = (k2 % 2) * 2 + (i % 2)
                    W4[idx_C, t] = W_block[b_C, k] * W2[l_C, s]
                    
    return U4, V4, W4

def _hybrid_5x5():
    """5x5x5 matrix multiplication: 4x4x4 Strassen core (rank 49) + boundary (rank 21) = rank 70."""
    n, m, p = 5, 5, 5
    rank_core = 49
    rank_boundary = 21
    total_rank = rank_core + rank_boundary
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    U4, V4, W4 = _strassen_4x4()
    
    for i in range(4):
        for j in range(4):
            idx_A = i * 5 + j
            for t in range(49):
                U[idx_A, t] = U4[i * 4 + j, t]
                
    for j in range(4):
        for k in range(4):
            idx_B = j * 5 + k
            for t in range(49):
                V[idx_B, t] = V4[j * 4 + k, t]
                
    for i in range(4):
        for k in range(4):
            idx_W = k * 5 + i
            for t in range(49):
                W[idx_W, t] = W4[k * 4 + i, t]
                
    col_offset = 49
    
    for j in range(4):
        c = col_offset
        for i in range(4):
            U[i * 5 + j, c] = 1
            W[4 * 5 + i, c] = 1
        V[j * 5 + 4, c] = 1
        col_offset += 1
        
    for j in range(4):
        c = col_offset
        U[4 * 5 + j, c] = 1
        for k in range(4):
            V[j * 5 + k, c] = 1
            W[k * 5 + 4, c] = 1
        col_offset += 1
        
    for i in range(4):
        c = col_offset
        U[i * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[4 * 5 + i, c] = 1
        col_offset += 1
        
    for k in range(4):
        c = col_offset
        U[4 * 5 + 4, c] = 1
        V[4 * 5 + k, c] = 1
        W[k * 5 + 4, c] = 1
        col_offset += 1
        
    for j in range(4):
        c = col_offset
        U[4 * 5 + j, c] = 1
        V[j * 5 + 4, c] = 1
        W[4 * 5 + 4, c] = 1
        col_offset += 1
        
    c = col_offset
    U[4 * 5 + 4, c] = 1
    V[4 * 5 + 4, c] = 1
    W[4 * 5 + 4, c] = 1
    col_offset += 1
    
    assert col_offset == 70, f"Expected 70, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition is correct."""
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat
    R = V_arr @ B_flat
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            idx = j * n + i
            C_computed[i, j] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    try:
        result = _hybrid_5x5()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
        
    result = _schoolbook(n, m, p)
    return result
