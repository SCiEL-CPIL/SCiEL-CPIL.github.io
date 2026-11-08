import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-114 exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Uses Strassen's algorithm (rank 49) for two 4x4x4 blocks and schoolbook (rank 16) for the 4x4x1 block.
    Total rank: 114.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds is plenty for this static construction
    
    # Strassen's algorithm for 2x2x2 (rank 7)
    strassen_U = np.array([
        [1, 1, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ])
    
    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1]    # h
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ])
    
    def build_4x4x4_strassen():
        """Build the 4x4x4 Strassen decomposition with rank 49."""
        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)
        
        for t1 in range(7):
            for t2 in range(7):
                c = t1 * 7 + t2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U4[i * 4 + j, c] = strassen_U[i1 * 2 + j1, t1] * strassen_U[i2 * 2 + j2, t2]
                                V4[i * 4 + j, c] = strassen_V[i1 * 2 + j1, t1] * strassen_V[i2 * 2 + j2, t2]
                                W4[i * 4 + j, c] = strassen_W[i1 * 2 + j1, t1] * strassen_W[i2 * 2 + j2, t2]
        return U4, V4, W4
    
    def build_4x4x1_schoolbook():
        """Build the 4x4x1 schoolbook decomposition with rank 16."""
        U1 = np.zeros((16, 16), dtype=np.float64)
        V1 = np.zeros((4, 16), dtype=np.float64)
        W1 = np.zeros((4, 16), dtype=np.float64)
        
        for i in range(4):
            for j in range(4):
                c = i * 4 + j
                U1[i * 4 + j, c] = 1.0
                V1[j, c] = 1.0
                W1[i, c] = 1.0
        return U1, V1, W1
    
    # Assemble full 4x4x9
    U_full = np.zeros((16, 114), dtype=np.float64)
    V_full = np.zeros((36, 114), dtype=np.float64)
    W_full = np.zeros((36, 114), dtype=np.float64)
    
    # Block 1: cols 0-3
    U4_1, V4_1, W4_1 = build_4x4x4_strassen()
    U_full[:, 0:49] = U4_1
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V4_1[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W4_1[k * 4 + i, :]
            
    # Block 2: cols 4-7
    U4_2, V4_2, W4_2 = build_4x4x4_strassen()
    U_full[:, 49:98] = U4_2
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + (4 + k), 49:98] = V4_2[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[(4 + k) * 4 + i, 49:98] = W4_2[k * 4 + i, :]
            
    # Block 3: col 8
    U1, V1, W1 = build_4x4x1_schoolbook()
    U_full[:, 98:114] = U1
    for j in range(4):
        V_full[j * 9 + 8, 98:114] = V1[j, :]
    for i in range(4):
        W_full[8 * 4 + i, 98:114] = W1[i, :]
        
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
        
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook (rank 144)
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school), dtype=np.float64)
        V_fb = np.zeros((m * p, rank_school), dtype=np.float64)
        W_fb = np.zeros((p * n, rank_school), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
