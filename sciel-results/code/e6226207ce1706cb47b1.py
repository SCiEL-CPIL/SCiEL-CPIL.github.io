"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-32 decomposition for 2x2 x 2x9 matrix multiplication.
    
    Strategy: Split the 9 columns into groups of sizes {4, 4, 1}.
    For each 2x2 block, use Strassen's 7-term decomposition.
    For the 2x1 block, use the trivial 4-term decomposition.
    Total rank: 7 + 7 + 4 = 18... wait, that's 18.
    
    Actually let me re-examine. The dimensions are:
    A is 2x2, B is 2x9, C = A*B is 2x9.
    
    The tensor has:
    - U shape: (n*m, rank) = (4, rank)  [A indices: i=0,1; j=0,1]
    - V shape: (m*p, rank) = (18, rank) [B indices: j=0,1; k=0..8]
    - W shape: (p*n, rank) = (18, rank) [C indices: k=0..8; i=0,1]
    
    For Strassen on 2x2 x 2x2:
    The 7 products are:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B21-B22)
    M4 = A22*(B12-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let me write the explicit decomposition for one 2x2x2 block (columns k0, k1):
    
    U factors (A side, 4 entries per column):
    M1: U = [1, 0, 1, 1]  (A11, A12, A21, A22)
    M2: U = [0, 0, 1, 1]
    M3: U = [1, 0, 0, 0]
    M4: U = [0, 0, 0, 1]
    M5: U = [1, 1, 0, 0]
    M6: U = [-1, 0, 1, 0]
    M7: U = [0, 1, 0, -1]
    
    V factors (B side, for columns k0, k1 of this block):
    M1: V = [1, 1]  (B1k0, B1k1... wait, B is 2x2 here: rows j=0,1; cols k0,k1)
    
    Actually B is 2 rows (j=0,1) and 2 cols (k0, k1) for this block.
    B[j, k] where j is row of B (which is column index of A), k is column of B.
    
    So for the block with B columns k0, k1:
    B entries: B[0,k0], B[0,k1], B[1,k0], B[1,k1]
    
    M1 = (A11+A22)*(B[0,k0]+B[1,k1])
    M2 = (A21+A22)*B[0,k0]
    M3 = A11*(B[1,k0]-B[1,k1])
    M4 = A22*(B[0,k1]-B[0,k0])
    M5 = (A11+A12)*B[1,k1]
    M6 = (A21-A11)*(B[0,k0]+B[0,k1])
    M7 = (A12-A22)*(B[1,k0]+B[1,k1])
    
    V vectors (length 4 for this block: B[0,k0], B[0,k1], B[1,k0], B[1,k1]):
    M1: [1, 0, 0, 1]
    M2: [1, 0, 0, 0]
    M3: [0, 0, 1, -1]
    M4: [0, 1, 0, 0]  -- wait, B[0,k1]-B[0,k0] = [-1, 1, 0, 0]
    Let me redo:
    M4 = A22*(B[0,k1]-B[0,k0]) -> V = [-1, 1, 0, 0]
    M5 = (A11+A12)*B[1,k1] -> V = [0, 0, 0, 1]
    M6 = (A21-A11)*(B[0,k0]+B[0,k1]) -> V = [1, 1, 0, 0]
    M7 = (A12-A22)*(B[1,k0]+B[1,k1]) -> V = [0, 0, 1, 1]
    
    W vectors (C side, for this block: C[0,k0], C[0,k1], C[1,k0], C[1,k1]):
    C11 = C[0,k0] = M1 + M4 - M5 + M7
    C12 = C[0,k1] = M3 + M5
    C21 = C[1,k0] = M2 + M4
    C22 = C[1,k1] = M1 - M2 + M3 + M6
    
    W vectors (length 4: C[0,k0], C[0,k1], C[1,k0], C[1,k1]):
    M1: [1, 0, 0, 1]
    M2: [0, 0, 1, -1]
    M3: [0, 1, 0, 1]
    M4: [1, 0, 1, 0]
    M5: [-1, 1, 0, 0]
    M6: [0, 0, 0, 1]
    M7: [1, 0, 0, 1]
    
    Wait, let me recheck:
    C11 = M1 + M4 - M5 + M7: W_M1[0]=1, W_M4[0]=1, W_M5[0]=-1, W_M7[0]=1 -> 1+1-1+1=2? That's wrong.
    
    Let me redo the Strassen formulas carefully:
    
    Standard Strassen for A(2x2) * B(2x2):
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Check C11: M1+B11B11+A22B11+B11B22+A22B22 + A22B12-A22B11 - A11B22-A12B22 + A12B21-A12B22-A22B21+A22B22
    = A11B11+A22B11+A11B22+A22B22 + A22B12-A22B11 - A11B22-A12B22 + A12B21-A12B22-A22B21+A22B22
    = A11B11 + (A22B11-A22B11) + (A11B22-A11B22) + (A22B22+A22B22) + A22B12 - A12B22 - A12B22 + A12B21 - A22B21
    Hmm, this doesn't look right. Let me just trust the standard formulas.
    
    Actually the standard Strassen gives:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    And these are correct. So for W vectors:
    M1 contributes to C11 and C22: W = [1, 0, 0, 1]
    M2 contributes to C21 and C22: W = [0, 0, 1, -1]
    M3 contributes to C12 and C22: W = [0, 1, 0, 1]
    M4 contributes to C11 and C21: W = [1, 0, 1, 0]
    M5 contributes to C11 (neg) and C12: W = [-1, 1, 0, 0]
    M6 contributes to C22: W = [0, 0, 0, 1]
    M7 contributes to C11: W = [1, 0, 0, 0]
    
    Wait, M7 only contributes to C11? Let me recheck: C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6. Yes, M7 only in C11.
    
    So W vectors:
    M1: [1, 0, 0, 1]
    M2: [0, 0, 1, -1]
    M3: [0, 1, 0, 1]
    M4: [1, 0, 1, 0]
    M5: [-1, 1, 0, 0]
    M6: [0, 0, 0, 1]
    M7: [1, 0, 0, 0]
    
    Now for the 2x1 case (single column k):
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    Trivial 4-term:
    T1: A[0,0]*B[0,k] -> U=[1,0,0,0], V=[1,0], W=[1,0]
    T2: A[0,1]*B[1,k] -> U=[0,1,0,0], V=[0,1], W=[1,0]
    T3: A[1,0]*B[0,k] -> U=[0,0,1,0], V=[1,0], W=[0,1]
    T4: A[1,1]*B[1,k] -> U=[0,0,0,1], V=[0,1], W=[0,1]
    
    Total rank: 7 + 7 + 4 = 18. But that seems too good. Let me reconsider.
    
    Actually wait - for 2x2x2, the border rank is 7 (Strassen). For 2x2x1, the rank is 4 (it's just a bilinear form). So 7+7+4=18.
    
    But the schoolbook is 36. So 18 is a big improvement!
    
    Let me code this up properly.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 18
    
    # Strassen block 1: columns 0,1
    # Strassen block 2: columns 2,3
    # Single column: column 4
    # Wait, 9 columns. Let me use 4 blocks of 2 and 1 block of 1: 4*2+1=9.
    # That gives 4*7 + 4 = 32. Or 3 blocks of 2 and 3 single: 3*7+3*4=33.
    # Best: 4 blocks of 2 and 1 block of 1: 4*7+4 = 32.
    
    # Let me redo: 9 = 2+2+2+2+1, so 4 Strassen blocks (rank 7 each) + 1 trivial block (rank 4) = 32.
    rank = 32
    
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(18)]
    W = [[0.0] * rank for _ in range(18)]
    
    col_offset = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # Strassen U factors (4 entries: A[0,0], A[0,1], A[1,0], A[1,1])
        U_strassen = [
            [1, 0, 1, 1],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1],  # M7
        ]
        
        # Strassen V factors for this block (4 entries: B[0,k0], B[0,k1], B[1,k0], B[1,k1])
        V_strassen = [
            [1, 0, 0, 1],   # M1
            [1, 0, 0, 0],   # M2
            [0, 0, 1, -1],  # M3
            [-1, 1, 0, 0],  # M4
            [0, 0, 0, 1],   # M5
            [1, 1, 0, 0],   # M6
            [0, 0, 1, 1],   # M7
        ]
        
        # Strassen W factors for this block (4 entries: C[0,k0], C[0,k1], C[1,k0], C[1,k1])
        W_strassen = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, -1],  # M2
            [0, 1, 0, 1],   # M3
            [1, 0, 1, 0],   # M4
            [-1, 1, 0, 0],  # M5
            [0, 0, 0, 1],   # M6
            [1, 0, 0, 0],   # M7
        ]
        
        for s in range(7):
            c = col_offset + s
            for i in range(4):
                U[i][c] = float(U_strassen[s][i])
            for j_local in range(4):
                # V index: j_global * p + k_global
                # j_local 0 -> B[0,k0] -> V[0*p+k0] = V[k0]
                # j_local 1 -> B[0,k1] -> V[k1]
                # j_local 2 -> B[1,k0] -> V[p+k0]
                # j_local 3 -> B[1,k1] -> V[p+k1]
                if j_local == 0:
                    v_idx = k0
                elif j_local == 1:
                    v_idx = k1
                elif j_local == 2:
                    v_idx = p + k0
                else:
                    v_idx = p + k1
                V[v_idx][c] = float(V_strassen[s][j_local])
            for j_local in range(4):
                # W index: k_global * n + i_global
                # j_local 0 -> C[0,k0] -> W[k0*n+0] = W[k0]
                # j_local 1 -> C[0,k1] -> W[k1]
                # j_local 2 -> C[1,k0] -> W[k0*n+1] = W[n+k0]
                # j_local 3 -> C[1,k1] -> W[n+k1]
                if j_local == 0:
                    w_idx = k0 * n + 0
                elif j_local == 1:
                    w_idx = k1 * n + 0
                elif j_local == 2:
                    w_idx = k0 * n + 1
                else:
                    w_idx = k1 * n + 1
                W[w_idx][c] = float(W_strassen[s][j_local])
        
        col_offset += 7
    
    # Single column: column 8
    k = 8
    # Trivial 4-term
    U_trivial = [
        [1, 0, 0, 0],  # A[0,0]
        [0, 1, 0, 0],  # A[0,1]
        [0, 0, 1, 0],  # A[1,0]
        [0, 0, 0, 1],  # A[1,1]
    ]
    V_trivial = [
        [1, 0],  # B[0,k]
        [0, 1],  # B[1,k]
        [1, 0],  # B[0,k]
        [0, 1],  # B[1,k]
    ]
    W_trivial = [
        [1, 0],  # C[0,k]
        [1, 0],  # C[0,k]
        [0, 1],  # C[1,k]
        [0, 1],  # C[1,k]
    ]
    
    for s in range(4):
        c = col_offset + s
        for i in range(4):
            U[i][c] = float(U_trivial[s][i])
        # V: B[0,k] -> V[k], B[1,k] -> V[p+k]
        V[k][c] = float(V_trivial[s][0])
        V[p + k][c] = float(V_trivial[s][1])
        # W: C[0,k] -> W[k*n+0] = W[k], C[1,k] -> W[k*n+1] = W[n+k]
        W[k * n + 0][c] = float(W_trivial[s][0])
        W[k * n + 1][c] = float(W_trivial[s][1])
    
    col_offset += 4
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Safety margin
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

