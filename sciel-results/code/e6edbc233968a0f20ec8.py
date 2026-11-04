import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# Pre-compute Strassen 2x2 matrices
U2 = np.array([
    [1, 1, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1]
], dtype=float)

V2 = np.array([
    [1, 1, 1, -1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1]
], dtype=float)

W2 = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 1, 0]
], dtype=float)

def _build_4x4x4_strassen():
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Big mults definitions
    # A, B, C are lists of (block_idx, sign)
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            # Vectorized update
            # U update
            for (bi, bj), sign_a in bm['A']:
                # U indices: I*4+J where I=2*bi+i2, J=2*bj+k2
                # U2 indices: i2*2+k2
                # This is a Kronecker-like structure
                # We can just loop, it's small
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

def _validate_numpy(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U444, V444, W444 = _build_4x4x4_strassen()
    
    if not _validate_numpy(U444, V444, W444, 4, 4, 4):
        return _schoolbook_fallback()
        
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1
    U[:, :49] = U444
    for k in range(4):
        for j in range(4):
            V[k*9+j, :49] = V444[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, :49] = W444[j*4+i, :]
            
    # Block 2
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), offset2:offset2+49] = V444[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, offset2:offset2+49] = W444[j*4+i, :]
            
    # Block 3
    offset3 = 98
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J, offset3+idx] = 1
        V[J*9+8, offset3+idx] = 1
        W[8*4+I, offset3+idx] = 1
        
    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
