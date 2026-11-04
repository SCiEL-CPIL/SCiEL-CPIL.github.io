import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Block decomposition with Strassen for 2x2x2 blocks.
    # A is 2x5, B is 5x6.
    # Split columns of A into 3 parts: 2, 2, 1.
    # Split rows of B into 3 parts: 2, 2, 1.
    # A = [A1, A2, A3] where A1, A2 are 2x2, A3 is 2x1.
    # B = [B1; B2; B3] where B1, B2 are 2x6, B3 is 1x6.
    # C = A1*B1 + A2*B2 + A3*B3.
    #
    # Term 1: A1 (2x2) * B1 (2x6).
    #   Split B1 into 3 2x2 blocks: B11, B12, B13.
    #   Use Strassen (rank 7) for each 2x2 * 2x2. Total rank 21.
    # Term 2: A2 (2x2) * B2 (2x6).
    #   Similar to Term 1. Total rank 21.
    # Term 3: A3 (2x1) * B3 (1x6).
    #   Outer product. Rank 2 (since 2*1*1=2, and it's essentially rank 2? No, 2x1 * 1x6 is rank 1 matrix?
    #   Wait. The tensor is T[i,k,r] = sum_j A[i,j] B[j,k].
    #   Here A is 2x1, B is 1x6.
    #   T[i,k,1] = A[i,1] * B[1,k].
    #   This can be decomposed into 2 terms?
    #   T = sum_r U[i,r] V[r,k].
    #   If rank 1: U[i,1]=A[i,1], V[1,k]=B[1,k]. Then U*V gives A[i,1]B[1,k]. Correct.
    #   So rank 1 is sufficient for 2x1 * 1x6.
    #   Total rank = 21 + 21 + 1 = 43.
    #
    # Can we do better?
    # Maybe split A into 2x1, 2x1, 2x1, 2x1, 2x1? No.
    # Maybe split B differently?
    # Let's stick to 43 first. It's a huge jump from 60.

    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists.
    # Shapes: U (10, R), V (30, R), W (12, R).
    # We will determine R dynamically.
    
    # Helper: Get Strassen U, V, W for 2x2 * 2x2.
    # Indices for U: (i, j) -> i*2+j. i in {0,1}, j in {0,1}.
    # Indices for V: (j, k) -> j*2+k. j in {0,1}, k in {0,1}.
    # Indices for W: (k, i) -> k*2+i. k in {0,1}, i in {0,1}.
    #
    # Standard Strassen decomposition (one variant):
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B00 - B01)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # Result C:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P2 + P3
    # C10 = P6 + P7
    # C11 = P1 - P3 + P4 + P5
    
    def get_strassen_factors():
        # Returns U, V, W for 2x2x2 multiplication.
        # U shape (4, 7), V shape (4, 7), W shape (4, 7).
        # U[i*2+j, r]
        # V[j*2+k, r]
        # W[k*2+i, r]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: P1
        # A: 00, 11
        U[0][0] = 1; U[3][0] = 1
        # B: 00, 11
        V[0][0] = 1; V[3][0] = 1
        # C: 00, 11
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: P2
        # A: 10, 11
        U[2][1] = 1; U[3][1] = 1
        # B: 00
        V[0][1] = 1
        # C: 01, 11
        W[1][1] = 1; W[3][1] = 1
        
        # r=2: P3
        # A: 00
        U[0][2] = 1
        # B: 01, -11
        V[1][2] = 1; V[3][2] = -1
        # C: 01, -11
        W[1][2] = 1; W[3][2] = -1
        
        # r=3: P4
        # A: 11
        U[3][3] = 1
        # B: 00, -01
        V[0][3] = 1; V[1][3] = -1
        # C: 00, 11
        W[0][3] = 1; W[3][3] = 1
        
        # r=4: P5
        # A: 00, 01
        U[0][4] = 1; U[1][4] = 1
        # B: 11
        V[3][4] = 1
        # C: -00, 11
        W[0][4] = -1; W[3][4] = 1
        
        # r=5: P6
        # A: 10, -00
        U[2][5] = 1; U[0][5] = -1
        # B: 00, 01
        V[0][5] = 1; V[1][5] = 1
        # C: 10
        W[2][5] = 1
        
        # r=6: P7
        # A: 01, -11
        U[1][6] = 1; U[3][6] = -1
        # B: 10, 11
        V[2][6] = 1; V[3][6] = 1
        # C: 00, 10
        W[0][6] = 1; W[2][6] = 1
        
        return U, V, W

    sU, sV, sW = get_strassen_factors()

    # Total Rank:
    # Block 1 (A1 B1): 3 sub-blocks * 7 = 21
    # Block 2 (A2 B2): 3 sub-blocks * 7 = 21
    # Block 3 (A3 B3): 1
    total_rank = 21 + 21 + 1
    R = total_rank
    
    U = [[0]*R for _ in range(n*m)] # 10 x 43
    V = [[0]*R for _ in range(m*p)] # 30 x 43
    W = [[0]*R for _ in range(p*n)] # 12 x 43
    
    col = 0
    
    # Helper to copy a block of factors into the main matrices
    def add_block(u_block, v_block, w_block, i_off, j_off, k_off):
        # i_off: offset for i in U, W
        # j_off: offset for j in U, V
        # k_off: offset for k in V, W
        nonlocal col
        for r in range(7):
            for u_i in range(4):
                # u_i is index in 2x2 block.
                # global index in U: (i_off + i_local)*2 + j_local?
                # No, U index is i*m+j.
                # In the 2x2 block, i_local, j_local in {0,1}.
                # Global i = i_off + i_local.
                # Global j = j_off + j_local.
                # So U index = (i_off + i_local)*2 + (j_off + j_local) -- WAIT.
                # U has shape (n*m, R). Index is i*m + j.
                # m=5.
                # So index = i*5 + j.
                
                # Let's decompose u_i (0..3) into i_l, j_l
                i_l = u_i // 2
                j_l = u_i % 2
                u_val = u_block[u_i][r]
                if u_val != 0:
                    gi = i_off + i_l
                    gj = j_off + j_l
                    U[gi*m + gj][col] = u_val
                    
            for v_i in range(4):
                # v_i in 2x2 block. j_l, k_l.
                j_l = v_i // 2
                k_l = v_i % 2
                v_val = v_block[v_i][r]
                if v_val != 0:
                    gj = j_off + j_l
                    gk = k_off + k_l
                    V[gj*p + gk][col] = v_val
                    
            for w_i in range(4):
                # w_i in 2x2 block. k_l, i_l.
                k_l = w_i // 2
                i_l = w_i % 2
                w_val = w_block[w_i][r]
                if w_val != 0:
                    gk = k_off + k_l
                    gi = i_off + i_l
                    W[gk*n + gi][col] = w_val
                    
            col += 1

    # Block 1: A1 (rows 0,1 of A? No, cols 0,1 of A) * B1 (rows 0,1 of B)
    # A is 2x5. A1 is cols 0,1. i in {0,1}, j in {0,1}.
    # B is 5x6. B1 is rows 0,1. j in {0,1}, k in {0..5}.
    # B1 is 2x6. Split into B11 (cols 0,1), B12 (cols 2,3), B13 (cols 4,5).
    
    # Sub-block 1.1: j in {0,1}, k in {0,1}
    add_block(sU, sV, sW, 0, 0, 0)
    
    # Sub-block 1.2: j in {0,1}, k in {2,3}
    add_block(sU, sV, sW, 0, 0, 2)
    
    # Sub-block 1.3: j in {0,1}, k in {4,5}
    add_block(sU, sV, sW, 0, 0, 4)
    
    # Block 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B)
    # A2: i in {0,1}, j in {2,3}.
    # B2: j in {2,3}, k in {0..5}.
    
    # Sub-block 2.1: j in {2,3}, k in {0,1}
    add_block(sU, sV, sW, 0, 2, 0)
    
    # Sub-block 2.2: j in {2,3}, k in {2,3}
    add_block(sU, sV, sW, 0, 2, 2)
    
    # Sub-block 2.3: j in {2,3}, k in {4,5}
    add_block(sU, sV, sW, 0, 2, 4)
    
    # Block 3: A3 (col 4 of A) * B3 (row 4 of B)
    # A3: i in {0,1}, j=4.
    # B3: j=4, k in {0..5}.
    # Rank 1.
    # U[gi*m + gj, col] = A[i, 4] -> 1 if gj=4.
    # V[gj*p + gk, col] = B[4, k] -> 1 if gj=4.
    # W[gk*n + gi, col] = C[k, i] -> 1.
    
    for gi in range(2):
        # U index: gi*5 + 4
        U[gi*m + 4][col] = 1
        
    for gk in range(6):
        # V index: 4*6 + gk
        V[4*p + gk][col] = 1
        
    for gi in range(2):
        for gk in range(6):
            # W index: gk*2 + gi
            W[gk*n + gi][col] = 1
            
    col += 1
    
    # Verification
    def verify():
        # Reconstruct T from U, V, W and compare with schoolbook.
        # T[i, k, r] = sum_c U[i*m+j? no, U is indexed by (i,j) implicitly? 
        # The problem statement says:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # "output factor follows the official trace convention W[k*n+i,r]"
        # This implies U is indexed by (i,j) as i*m+j?
        # And V by (j,k) as j*p+k?
        # And W by (k,i) as k*n+i?
        # Yes, standard tensor decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is not right.
        # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
        # Tensor T[i,j,k] has components A[i,j]B[j,k]? No, T is the multiplication tensor.
        # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # Decomposition: C[i,k] = sum_r U[i,j,r]? No.
        # Usually: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] ... no.
        # The standard form is:
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] is for 3-tensors.
        # But here we have matrix multiplication which is a 3-tensor T[i,j,k] (if we flatten indices?).
        # Actually, the task says:
        # U: (n*m, rank) -> indexed by (i,j)
        # V: (m*p, rank) -> indexed by (j,k)
        # W: (p*n, rank) -> indexed by (k,i)
        # And the condition is:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k}? No.
        # It must be that:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if the index combination corresponds to a valid multiplication term?
        # No, the tensor T for matrix multiplication is defined such that:
        # T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to get C[i,k].
        # The decomposition says:
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
        # If so, U, V, W are 3-tensors? But they are given as 2D matrices.
        # Ah, the indices in U, V, W are flattened.
        # U[ (i,j), r ]
        # V[ (j,k), r ]
        # W[ (k,i), r ]
        # The condition is:
        # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k?
        # No, that would imply the tensor is all ones.
        # The matrix multiplication tensor T has T[i, j, k] = 1 for all i, j, k?
        # No, T maps A, B to C.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # This is a linear map.
        # The tensor T is defined on indices (i, j, k) such that:
        # <T, A \otimes B> = C.
        # T[i, j, k] = 1. (Since A[i,j] B[j,k] contributes to C[i,k]).
        # Wait, if T[i,j,k]=1, then sum_r U[i,j,r]V[j,k,r]W[k,i,r] = 1?
        # Let's check the seed program.
        # Seed: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # For a specific i,j,k, there is exactly one c (schoolbook) where U,V,W are 1.
        # For other c, they are 0.
        # So sum_r ... = 1.
        # Yes, the tensor is the "all-ones" tensor on the index sets?
        # No, the indices are linked.
        # The condition is:
        # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
        
        # Let's verify this condition.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    if not verify():
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
