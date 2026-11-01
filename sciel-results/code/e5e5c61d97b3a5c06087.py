import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    
    # Total rank: 7 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for B (2x2 block): 0:00, 1:01, 2:10, 3:11 (relative to block)
    # Indices for C (2x2 block): 0:00, 1:01, 2:10, 3:11 (relative to block)
    
    strassen_U = [
        [1, 0, 1, 0, 0, 0, 0], # M1: A00, A11
        [0, 1, 0, 0, 0, 0, 0], # M2: A10, A11 -> Wait, M2 is (A10+A11)B00. U should be A10, A11.
        # Let's re-verify M2. M2 = (A10 + A11) B00.
        # So U for M2 is A10=1, A11=1.
        # My previous list was:
        # M1: A00=1, A11=1
        # M2: A10=1, A11=1
        # M3: A00=1
        # M4: A11=1
        # M5: A00=1, A01=1
        # M6: A10=1, A00=-1
        # M7: A01=1, A11=-1
        
        # Let's list them clearly.
    ]
    
    # Correct Strassen U vectors (indices 0..3 for A00, A01, A10, A11)
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Correct Strassen V vectors (indices 0..3 for B00, B01, B10, B11)
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Correct Strassen W vectors (indices 0..3 for C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # 7 blocks of 2x2
    for b in range(7):
        # B columns: 2b, 2b+1
        # C columns: 2b, 2b+1
        for s in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    idx_u = i * m + j
                    U[idx_u][col] = U_strassen[s][i * 2 + j]
            
            # V
            for j in range(2):
                for k in range(2):
                    idx_v = j * p + (2 * b + k)
                    V[idx_v][col] = V_strassen[s][j * 2 + k]
            
            # W
            for k in range(2):
                for i in range(2):
                    idx_w = (2 * b + k) * n + i
                    W[idx_w][col] = W_strassen[s][k * 2 + i]
            
            col += 1
    
    # 1 block of 2x1 (last column, index 14)
    # C0,14 = A0,0 B0,14 + A0,1 B1,14
    # C1,14 = A1,0 B0,14 + A1,1 B1,14
    
    # T1: A00 B0,14 -> C0,14
    U[0][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 0][col] = 1
    col += 1
    
    # T2: A01 B1,14 -> C0,14
    U[1][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 0][col] = 1
    col += 1
    
    # T3: A10 B0,14 -> C1,14
    U[2][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 1][col] = 1
    col += 1
    
    # T4: A11 B1,14 -> C1,14
    U[3][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
