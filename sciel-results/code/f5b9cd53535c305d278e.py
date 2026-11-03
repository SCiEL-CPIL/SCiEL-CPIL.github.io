# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    total_rank = 0
    U = []
    V = []
    W = []
    
    def add_block(block_U, block_V, block_W, rA1, rA2, cA1, cA2, rB1, rB2, cB1, cB2, rC1, rC2, cC1, cC2):
        nonlocal total_rank, U, V, W
        rank_block = len(block_U[0])
        n_A = rA2 - rA1
        m_A = cA2 - cA1
        n_B = rB2 - rB1
        m_B = cB2 - cB1
        n_C = rC2 - rC1
        m_C = cC2 - cC1
        
        # Expand U, V, W
        for r in range(rank_block):
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for i in range(n_A):
                for j in range(m_A):
                    val = block_U[i * m_A + j][r]
                    if val != 0:
                        U_col[(rA1 + i) * m + (cA1 + j)] = val
                        
            for j in range(n_B):
                for k in range(m_B):
                    val = block_V[j * m_B + k][r]
                    if val != 0:
                        V_col[(rB1 + j) * p + (cB1 + k)] = val
                        
            for k in range(n_C):
                for i in range(m_C):
                    val = block_W[k * m_C + i][r]
                    if val != 0:
                        W_col[(rC1 + k) * n + (cC1 + i)] = val
                        
            U.append(U_col)
            V.append(V_col)
            W.append(W_col)
            
        total_rank += rank_block

    # 1. A22 B22 -> C22 (2x2x2, Strassen Rank 7)
    # Strassen Terms
    # U, V are 4x7, W is 4x7
    sU = [
        [1, 0, 1, 0, 1, 0, 0], # A00, A11
        [0, 0, 0, 1, 0, 0, 0], # A10? No, A11 is index 3.
        # Let's use 0-indexed local:
        # 0:00, 1:01, 2:10, 3:11
        # M1: A00+A11 -> [1, 0, 0, 1]
        # M2: A10+A11 -> [0, 0, 1, 1]
        # M3: A00 -> [1, 0, 0, 0]
        # M4: A11 -> [0, 0, 0, 1]
        # M5: A00+A01 -> [1, 1, 0, 0]
        # M6: A10-A00 -> [-1, 0, 1, 0]
        # M7: A01-A11 -> [0, 1, 0, -1]
        [1, 0, 0, 1, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1]
    ]
    sV = [
        [1, 0, 0, 0, 0, 1, 0], # B00, B11 for M1?
        # M1: B00+B11 -> [1, 0, 0, 1]
        # M2: B00 -> [1, 0, 0, 0]
        # M3: B01-B11 -> [0, 1, 0, -1]
        # M4: B10-B00 -> [-1, 0, 1, 0]
        # M5: B11 -> [0, 0, 0, 1]
        # M6: B00+B01 -> [1, 1, 0, 0]
        # M7: B10+B11 -> [0, 0, 1, 1]
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1]
    ]
    sW = [
        [1, 0, 0, 1, 0, 0, 0], # C00
        [0, 1, 0, 0, 1, 0, 0], # C01
        [0, 0, 0, 1, 0, 0, 0], # C10
        [0, 0, 0, 0, 0, 0, 1]  # C11
        # Check M1: C00(1), C11(1), C01(-1), C10(1)
        # sW[0][0] (C00) = 1. sW[3][0] (C11) = 1. sW[1][0] (C01) = 0? 
        # Wait, my sW construction above is just placeholders. I need to derive them.
    ]
    
    # Correct Strassen W derivation:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Col 0 (M1): C00=1, C01=0, C10=0, C11=1
    # Col 1 (M2): C00=0, C01=0, C10=1, C11=-1
    # Col 2 (M3): C00=0, C01=1, C10=0, C11=1
    # Col 3 (M4): C00=1, C01=0, C10=1, C11=0
    # Col 4 (M5): C00=-1, C01=1, C10=0, C11=0
    # Col 5 (M6): C00=0, C01=0, C10=0, C11=1
    # Col 6 (M7): C00=1, C01=0, C10=0, C11=0
    
    sW_correct = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    add_block(sU, sV, sW_correct, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2)

    # Helper for Schoolbook
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # 2. A23 B32 -> C22 (2x3x2)
    U2, V2, W2 = get_schoolbook(2, 3, 2)
    add_block(U2, V2, W2, 0, 2, 2, 5, 2, 5, 0, 2, 0, 2, 0, 2)

    # 3. A22 B23 -> C23 (2x2x3)
    U3, V3, W3 = get_schoolbook(2, 2, 3)
    add_block(U3, V3, W3, 0, 2, 0, 2, 0, 2, 2, 5, 2, 5, 0, 2)

    # 4. A23 B33 -> C23 (2x3x3)
    U4, V4, W4 = get_schoolbook(2, 3, 3)
    add_block(U4, V4, W4, 0, 2, 2, 5, 2, 5, 2, 5, 2, 5, 0, 2)

    # 5. A32 B22 -> C32 (3x2x2)
    U5, V5, W5 = get_schoolbook(3, 2, 2)
    add_block(U5, V5, W5, 2, 5, 0, 2, 0, 2, 0, 2, 0, 2, 2, 5)

    # 6. A33 B32 -> C32 (3x3x2)
    U6, V6, W6 = get_schoolbook(3, 3, 2)
    add_block(U6, V6, W6, 2, 5, 2, 5, 2, 5, 0, 2, 0, 2, 2, 5)

    # 7. A32 B23 -> C33 (3x2x3)
    U7, V7, W7 = get_schoolbook(3, 2, 3)
    add_block(U7, V7, W7, 2, 5, 0, 2, 0, 2, 2, 5, 2, 5, 2, 5)

    # 8. A33 B33 -> C33 (3x3x3)
    U8, V8, W8 = get_schoolbook(3, 3, 3)
    add_block(U8, V8, W8, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
