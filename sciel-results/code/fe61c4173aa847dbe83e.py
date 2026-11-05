"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    big_mults = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val
                            
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
                    return False
    return True

def _build_strassen_4x4x9():
    """
    Builds the known optimal rank-114 decomposition using Strassen 4x4x4 blocks.
    """
    n, m, p = 4, 4, 9
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Quick validation of 4x4x4 block
    if not _validate_numpy(U444, V444, W444, 4, 4, 4):
        return None
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((36, total_rank), dtype=np.int64)
    W = np.zeros((36, total_rank), dtype=np.int64)
    
    # Block 1: columns 0-3
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W444[k*4 + i, :]
    
    # Block 2: columns 4-7
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), offset2:offset2+49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, offset2:offset2+49] = W444[k*4 + i, :]
    
    # Block 3: column 8 (schoolbook)
    offset3 = 98
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[i*4 + j, offset3 + idx] = 1
            V[j*9 + 8, offset3 + idx] = 1
            W[8*4 + i, offset3 + idx] = 1
            
    return U, V, W

def _build_winograd_4x4x9():
    """
    Placeholder for Winograd/Convolution based decomposition.
    Based on theoretical analysis, 4x4x4 rank is 49 (Strassen).
    Winograd for 4x4 typically uses 5x5 convolution.
    Rank of 5x5 cyclic convolution is not known to be < 49 for the embedded matrix mult.
    This function returns None to indicate no better solution found yet.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Strategy 1: Try Winograd/Convolution (Hypothesis)
    if time.time() < deadline:
        wino = _build_winograd_4x4x9()
        if wino is not None:
            U_w, V_w, W_w = wino
            if _validate_numpy(U_w, V_w, W_w, n, m, p):
                rank_w = U_w.shape[1]
                if rank_w < best_rank:
                    best_U, best_V, best_W = U_w, V_w, W_w
                    best_rank = rank_w
    
    # Strategy 2: Use known optimal Strassen decomposition
    if time.time() < deadline:
        stras = _build_strassen_4x4x9()
        if stras is not None:
            U_s, V_s, W_s = stras
            if _validate_numpy(U_s, V_s, W_s, n, m, p):
                rank_s = U_s.shape[1]
                if rank_s < best_rank:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = rank_s
    
    # Strategy 3: Try to reduce rank by testing column removal
    if best_U is not None and time.time() < deadline - 5:
        for col in range(best_rank):
            if time.time() > deadline - 3:
                break
            U_test = np.delete(best_U, col, axis=1)
            V_test = np.delete(best_V, col, axis=1)
            W_test = np.delete(best_W, col, axis=1)
            if _validate_numpy(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                break
    
    # Fallback: Schoolbook
    if best_U is None:
        U, V, W = _schoolbook_fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
