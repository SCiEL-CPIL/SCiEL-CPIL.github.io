import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 2, 2, 16
    # 4 independent 2x2 blocks in the p dimension
    num_blocks = p // 2  # = 8 blocks of size 1? No, p=16, so 8 pairs? 
    # Actually, the 2x16 matrix has 16 columns. We group them into 8 pairs of 2 columns each.
    # Each pair forms a 2x2 sub-matrix.
    # So we have 8 independent 2x2 matrix multiplications.
    
    # Wait, let me reconsider. A is 2x2, B is 2x16.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each k (0..15), C[:,k] = A * B[:,k]
    # Each B[:,k] is a 2x1 vector, so this is a 2x2 times 2x1, giving 2x1.
    # That's not a 2x2 multiplication.
    
    # Alternatively, group columns: for k=0,1: C[:,0:2] = A * B[:,0:2] where B[:,0:2] is 2x2.
    # So we have 8 independent 2x2 x 2x2 multiplications.
    # Each has rank 7 (Strassen), so total rank = 8 * 7 = 56.
    
    # That's better than 64! Let me implement Strassen for 8 blocks.
    
    rank = 8 * 7  # 56
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen A-coefficients (for each of 7 terms): which A entries contribute with what coefficient
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_A = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    
    # Strassen B-coefficients: which B entries contribute
    # B indices for a 2x2 block: 0=B00, 1=B01, 2=B10, 3=B11
    strassen_B = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10 - B11
        [1, 0, -1, 0], # M4: B00 - B10
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    # Output coefficients: for each of 4 output entries, which M terms contribute with what coefficient
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_C = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],  # C11 = M1 - M2 + M3 + M6
    ]
    
    for block in range(8):
        for s in range(7):  # 7 Strassen terms per block
            col = block * 7 + s
            # U: A coefficients. A is 2x2, indexed as i*m+j = i*2+j
            for ai in range(4):
                i = ai // 2
                j = ai % 2
                U[i * m + j][col] = strassen_A[s][ai]
            
            # V: B coefficients. B is 2x16, for this block, columns are block*2 and block*2+1
            # B indexed as j*p+k = j*16+k
            for bi in range(4):
                bj = bi // 2  # row of B (0 or 1)
                bk = bi % 2   # column within block (0 or 1)
                k_global = block * 2 + bk
                V[bj * p + k_global][col] = strassen_B[s][bi]
            
            # W: C coefficients. C is 2x16, indexed as k*n+i = k*2+i
            for ci in range(4):
                ck = ci // 2  # column within block (0 or 1)
                ci_row = ci % 2  # row of C (0 or 1)
                k_global = block * 2 + ck
                W[k_global * n + ci_row][col] = strassen_C[ci][s]
    
    return {"U": U, "V": V, "W": W}
