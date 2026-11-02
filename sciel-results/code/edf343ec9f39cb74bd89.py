def _recursive_strassen(n, m, p):
    """
    Generate a rank-(7^k) decomposition for n x m by m x p matrix multiplication
    using recursive Strassen. Returns (U, V, W) as lists of lists of Fractions.
    
    For n=m=p=4, this gives rank 49.
    The tensor indices follow: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 2 and m == 2 and p == 2:
        # Standard Strassen for 2x2x2, rank 7
        # Indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C (W): index = k*2 + i
        # 0: i=0,k=0 (C00)
        # 1: i=1,k=0 (C10)
        # 2: i=0,k=1 (C01)
        # 3: i=1,k=1 (C11)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)*(B11+B22)
        # A: 0, 3. B: 0, 3.
        # C: C00 (0), C11 (3)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)*B11
        # A: 2, 3. B: 0.
        # C: C10 (1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # M3 = A11*(B11-B21)
        # A: 0. B: 0, 2.
        # C: C00 (0), C01 (2) with -1
        U[0][2] = 1
        V[0][2] = 1; V[2][2] = -1
        W[0][2] = 1; W[2][2] = -1
        
        # M4 = A22*(B22-B12)
        # A: 3. B: 3, 1.
        # C: C11 (3), C10 (1) with -1
        U[3][3] = 1
        V[3][3] = 1; V[1][3] = -1
        W[3][3] = 1; W[1][3] = -1
        
        # M5 = (A11+A12)*B22
        # A: 0, 1. B: 3.
        # C: C01 (2)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        # A: 2, 0. B: 0, 1.
        # C: C10 (1), C11 (3)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1; W[3][5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        # A: 1, 3. B: 2, 3.
        # C: C01 (2), C11 (3)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[2][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    # For 4x4x4: use 2-level Strassen
    # Split into four 2x2 submatrices
    # We'll combine the 2x2x2 Strassen decompositions
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _recursive_strassen(2, 2, 2)
    
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # The 7 block-level Strassen terms
    # M1: A blocks {0,3} with coeffs {1,1}, B blocks {0,3} with coeffs {1,1}
    # M2: A blocks {2,3} with coeffs {1,1}, B blocks {0} with coeff {1}
    # M3: A blocks {0} with coeff {1}, B blocks {0,2} with coeffs {1,-1}
    # M4: A blocks {3} with coeff {1}, B blocks {3,1} with coeffs {1,-1}
    # M5: A blocks {0,1} with coeffs {1,1}, B blocks {3} with coeff {1}
    # M6: A blocks {2,0} with coeffs {1,-1}, B blocks {0,1} with coeffs {1,1}
    # M7: A blocks {1,3} with coeffs {1,-1}, B blocks {2,3} with coeffs {1,1}
    
    block_A = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, 1)],   # M2
        [(0, 1)],           # M3
        [(3, 1)],           # M4
        [(0, 1), (1, 1)],   # M5
        [(2, 1), (0, -1)],  # M6
        [(1, 1), (3, -1)],  # M7
    ]
    block_B = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(0, 1), (2, -1)],  # M3
        [(3, 1), (1, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)],   # M7
    ]
    block_C = [
        [(0, 1), (3, 1)],   # M1 -> C00 += M1, C11 += M1 (Blocks 0 and 3)
        [(2, 1)],           # M2 -> C10 += M2 (Block 2)
        [(0, 1), (1, -1)],  # M3 -> C00 += M3, C01 -= M3 (Blocks 0 and 1)
        [(3, 1), (2, -1)],  # M4 -> C11 += M4, C10 -= M4 (Blocks 3 and 2)
        [(1, 1)],           # M5 -> C01 += M5 (Block 1)
        [(2, 1), (3, 1)],   # M6 -> C10 += M6, C11 += M6 (Blocks 2 and 3)
        [(1, 1), (3, 1)],   # M7 -> C01 += M7, C11 += M7 (Blocks 1 and 3)
    ]
    
    for t in range(7):
        for s in range(7):
            rank_idx = t * 7 + s
            
            # A side
            for bi, coeff in block_A[t]:
                for i_loc in range(2):
                    for j_loc in range(2):
                        i_global = 2 * (bi // 2) + i_loc
                        j_global = 2 * (bi % 2) + j_loc
                        a_idx = i_global * 4 + j_global
                        
                        val = U2[i_loc * 2 + j_loc][s]
                        if val != 0:
                            U[a_idx][rank_idx] += coeff * val
            
            # B side
            for bj, coeff in block_B[t]:
                for j_loc in range(2):
                    for k_loc in range(2):
                        j_global = 2 * (bj // 2) + j_loc
                        k_global = 2 * (bj % 2) + k_loc
                        b_idx = j_global * 4 + k_global
                        
                        val = V2[j_loc * 2 + k_loc][s]
                        if val != 0:
                            V[b_idx][rank_idx] += coeff * val
            
            # C side
            for bc, coeff in block_C[t]:
                for k_loc in range(2):
                    for i_loc in range(2):
                        k_global = 2 * (bc // 2) + k_loc
                        i_global = 2 * (bc % 2) + i_loc
                        c_idx = k_global * 4 + i_global
                        
                        val = W2[k_loc * 2 + i_loc][s]
                        if val != 0:
                            W[c_idx][rank_idx] += coeff * val
    
    return U, V, W
