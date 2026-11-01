import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Outer dimensions: n/2, m/2, p/2
    # n=2 -> 1 block
    # m=4 -> 2 blocks
    # p=6 -> 3 blocks
    
    # Strassen 2x2x2 decomposition
    # Indices for 2x2 matrix: 00, 01, 10, 11
    # A indices: 0, 1, 2, 3
    # B indices: 0, 1, 2, 3
    # C indices: 0, 1, 2, 3
    
    # Strassen coefficients for U, V, W
    # U is for A, V for B, W for C
    # r in 0..6
    
    U_strassen = np.zeros((4, 7), dtype=float)
    V_strassen = np.zeros((4, 7), dtype=float)
    W_strassen = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    # Note: Standard Strassen indices
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # M1
    U_strassen[0, 0] += 1; U_strassen[3, 0] += 1
    V_strassen[0, 0] += 1; V_strassen[3, 0] += 1
    W_strassen[0, 0] += 1; W_strassen[3, 0] += 1
    
    # M2
    U_strassen[2, 1] += 1; U_strassen[3, 1] += 1
    V_strassen[0, 1] += 1
    W_strassen[2, 1] += 1
    
    # M3
    U_strassen[0, 2] += 1
    V_strassen[1, 2] += 1; V_strassen[3, 2] -= 1
    W_strassen[1, 2] += 1
    
    # M4
    U_strassen[3, 3] += 1
    V_strassen[2, 3] += 1; V_strassen[0, 3] -= 1
    W_strassen[0, 3] += 1; W_strassen[2, 3] += 1
    
    # M5
    U_strassen[0, 4] += 1; U_strassen[1, 4] += 1
    V_strassen[3, 4] += 1
    W_strassen[0, 4] -= 1; W_strassen[1, 4] += 1
    
    # M6
    U_strassen[2, 5] += 1; U_strassen[0, 5] -= 1
    V_strassen[0, 5] += 1; V_strassen[1, 5] += 1
    W_strassen[3, 5] += 1
    
    # M7
    U_strassen[1, 6] += 1; U_strassen[3, 6] -= 1
    V_strassen[2, 6] += 1; V_strassen[3, 6] += 1
    W_strassen[0, 6] += 1; W_strassen[3, 6] += 1
    
    # Outer decomposition 1x2, 2x3
    # A blocks: 1 row (I=0), 2 cols (J=0,1)
    # B blocks: 2 rows (J=0,1), 3 cols (K=0,1,2)
    # C blocks: 1 row (I=0), 3 cols (K=0,1,2)
    
    # Rank 6
    U_outer = np.zeros((1*2, 6), dtype=float) # 1x2 = 2 elements
    V_outer = np.zeros((2*3, 6), dtype=float) # 2x3 = 6 elements
    W_outer = np.zeros((3*1, 6), dtype=float) # 3x1 = 3 elements (Wait, W shape is p*n = 6*2=12? No, outer W is p/2 * n/2 = 3*1=3)
    
    # Outer U indices: I*m/2 + J = 0*2 + J = J
    # Outer V indices: J*p/2 + K = J*3 + K
    # Outer W indices: K*n/2 + I = K*1 + 0 = K
    
    for r_out in range(6):
        J = r_out // 3
        K = r_out % 3
        
        U_outer[J, r_out] = 1
        V_outer[J*3 + K, r_out] = 1
        W_outer[K, r_out] = 1
        
    # Combine
    rank = 7 * 6
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Mapping global indices to block indices
    # A: n x m -> 1 x 2 blocks of 2x2
    # A[i, j] -> block I=0, J=j//2, local i%2, j%2
    # Global U index: i*m + j
    # Block U index: I*(m/2) + J = J
    # Local U index: (i%2)*2 + (j%2)
    
    # B: m x p -> 2 x 3 blocks of 2x2
    # B[i, j] -> block J=i//2, K=j//2, local i%2, j%2
    # Global V index: i*p + j
    # Block V index: J*(p/2) + K = J*3 + K
    # Local V index: (i%2)*2 + (j%2)
    
    # C: n x p -> 1 x 3 blocks of 2x2
    # C[i, j] -> block I=0, K=j//2, local i%2, j%2
    # Global W index: j*n + i (Contract: W[k*n+i, r])
    # Block W index: K*(n/2) + I = K*1 + 0 = K
    # Local W index: (i%2)*2 + (j%2)
    
    for r_out in range(6):
        for r_in in range(7):
            r = r_out * 7 + r_in
            
            # U
            for i in range(n):
                for j in range(m):
                    I = 0
                    J = j // 2
                    li = i % 2
                    lj = j % 2
                    local_u = li * 2 + lj
                    block_u = J
                    global_u = i * m + j
                    U[global_u, r] = U_outer[block_u, r_out] * U_strassen[local_u, r_in]
            
            # V
            for i in range(m):
                for j in range(p):
                    J = i // 2
                    K = j // 2
                    li = i % 2
                    lj = j % 2
                    local_v = li * 2 + lj
                    block_v = J * 3 + K
                    global_v = i * p + j
                    V[global_v, r] = V_outer[block_v, r_out] * V_strassen[local_v, r_in]
            
            # W
            for i in range(n):
                for j in range(p):
                    I = 0
                    K = j // 2
                    li = i % 2
                    lj = j % 2
                    local_w = li * 2 + lj
                    block_w = K
                    global_w = j * n + i
                    W[global_w, r] = W_outer[block_w, r_out] * W_strassen[local_w, r_in]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
