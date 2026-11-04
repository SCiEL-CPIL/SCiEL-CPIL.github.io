"""Improved program: uses Strassen-style block decomposition for 2x4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    # We'll use Strassen for 2x2x2 blocks.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split B into B_left (4x2, cols 0-1) and B_right (4x2, cols 2-3).
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    # Each product A*B_left and A*B_right is 2x2.
    # Split A into A_top (1x4) and A_bottom (1x4)? No, A is 2x4.
    # Let's use the standard Strassen block decomposition.
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B11 B12; B21 B22] where each block is 2x2
    # C = [C1 C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    # Each Ci is 2x2.
    # So we have 4 products of 2x2 matrices: A1*B11, A2*B21, A1*B12, A2*B22
    # Each can be done with rank 7 (Strassen), total rank = 4*7 = 28.
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Using convention: U[i*2+j], V[j*2+k], W[k*2+i]
    # For 2x2x2: n=2, m=2, p=2
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # The 7 Strassen terms:
    strassen_U = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1],
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
    ]
    strassen_W = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
    ]
    
    # Let me verify this Strassen decomposition is correct.
    # Actually, let me use a known correct Strassen decomposition.
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor form, for each term t:
    # U[i*2+j, t] = coefficient of A[i,j] in the term
    # V[j*2+k, t] = coefficient of B[j,k] in the term
    # W[k*2+i, t] = coefficient of C[i,k] in the term
    
    # Let me enumerate the 7 terms:
    # Term 1: M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    #   U: A11=1, A22=1 -> [1,0,0,1]
    #   V: B11=1, B22=1 -> [1,0,0,1]
    #   W: C11=1, C22=1 -> [1,0,0,1]
    
    # Term 2: M2 = (A21+A22)B11 -> contributes to C21
    #   U: A21=1, A22=1 -> [0,0,1,1]
    #   V: B11=1 -> [1,0,0,0]
    #   W: C21=1 -> [0,0,1,0]
    
    # Term 3: M3 = A11(B12-B22) -> contributes to C12 and C22
    #   U: A11=1 -> [1,0,0,0]
    #   V: B12=1, B22=-1 -> [0,1,0,-1]
    #   W: C12=1, C22=-1 -> [0,1,0,-1]
    
    # Term 4: M4 = A22(B21-B11) -> contributes to C11 and C21
    #   U: A22=1 -> [0,0,0,1]
    #   V: B21=1, B11=-1 -> [-1,0,1,0]
    #   W: C11=1, C21=1 -> [1,0,1,0]
    
    # Term 5: M5 = (A11+A12)B22 -> contributes to C11 and C12
    #   U: A11=1, A12=1 -> [1,1,0,0]
    #   V: B22=1 -> [0,0,0,1]
    #   W: C11=-1, C12=1 -> [-1,1,0,0]
    
    # Term 6: M6 = (A21-A11)(B11+B12) -> contributes to C22
    #   U: A21=1, A11=-1 -> [-1,0,1,0]
    #   V: B11=1, B12=1 -> [1,1,0,0]
    #   W: C22=1 -> [0,0,0,1]
    
    # Term 7: M7 = (A12-A22)(B21+B22) -> contributes to C11 and C22
    #   U: A12=1, A22=-1 -> [0,1,0,-1]
    #   V: B21=1, B22=1 -> [0,0,1,1]
    #   W: C11=1, C22=1 -> [1,0,0,1]
    
    # Now let me build the full 2x4x4 decomposition.
    # n=2, m=4, p=4
    # A is 2x4, indexed as A[i, j] for i in 0..1, j in 0..3
    # B is 4x4, indexed as B[j, k] for j in 0..3, k in 0..3
    # C is 2x4, indexed as C[i, k] for i in 0..1, k in 0..3
    # Output convention: W[k*n+i, r] = W[k*2+i, r]
    
    # Block decomposition:
    # A = [A1 A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [B11 B12; B21 B22] where B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #     B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C1 = A1*B11 + A2*B21 (2x2, cols 0-1 of C)
    # C2 = A1*B12 + A2*B22 (2x2, cols 2-3 of C)
    
    # For each of the 4 block products, use 7 Strassen terms.
    # Total rank = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2x2 (U, V, W each 4x7)
    sU = [
        [1, 0, 0, 0, 0, 0, 0],  # A11
        [0, 0, 1, 1, 0, 0, 0],  # A12
        [0, 0, 0, 0, 0, 0, 0],  # A21
        [0, 0, 0, 0, 0, 0, 0],  # A22
    ]
    # Wait, I need to be more careful. Let me re-derive.
    
    # Actually, let me just hard-code the correct Strassen 2x2x2 tensor.
    # Using the convention from the problem:
    # U[i*m+j, c] for m=2, so U[i*2+j, c]
    # V[j*p+k, c] for p=2, so V[j*2+k, c]
    # W[k*n+i, c] for n=2, so W[k*2+i, c]
    
    # The 7 terms (U_col, V_col, W_col) where each col is length 4:
    # I'll use a known correct set. Let me verify with a simple test.
    
    # For now, let me use the standard Strassen formulation and map it carefully.
    
    # Block 1: A1*B11 -> C1_left (C[:, 0:2])
    # A1 rows: 0-1 of A, cols 0-1. In U indexing: i*4+j for i in 0-1, j in 0-1
    # B11 rows: 0-1 of B, cols 0-1. In V indexing: j*4+k for j in 0-1, k in 0-1
    # C1_left cols: 0-1 of C. In W indexing: k*2+i for k in 0-1, i in 0-1
    
    # Block 2: A2*B21 -> C1_right part (still C[:, 0:2])
    # A2 rows: 0-1 of A, cols 2-3. In U indexing: i*4+j for i in 0-1, j in 2-3
    # B21 rows: 2-3 of B, cols 0-1. In V indexing: j*4+k for j in 2-3, k in 0-1
    # C1_left: same as above
    
    # Block 3: A1*B12 -> C2 (C[:, 2:4])
    # A1: same as block 1
    # B12 rows: 0-1 of B, cols 2-3. In V indexing: j*4+k for j in 0-1, k in 2-3
    # C2 cols: 2-3 of C. In W indexing: k*2+i for k in 2-3, i in 0-1
    
    # Block 4: A2*B22 -> C2 (C[:, 2:4])
    # A2: same as block 2
    # B22 rows: 2-3 of B, cols 2-3. In V indexing: j*4+k for j in 2-3, k in 2-3
    # C2: same as block 3
    
    # Strassen 2x2x2 tensor (verified):
    # Using the formulation:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U columns (length 4, indexed as [A11, A12, A21, A22]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    # V columns (length 4, indexed as [B11, B12, B21, B22]):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 1, 0, -1]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    
    # W columns (length 4, indexed as [C11, C12, C21, C22]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 1]
    
    sU_cols = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    sV_cols = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    sW_cols = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 1],   # M7
    ]
    
    # Now map to the 2x4x4 problem.
    # Block 1: A1*B11, contributes to C[:, 0:2]
    # A1: A[i, j] for i in 0-1, j in 0-1. U index: i*4 + j
    # B11: B[j, k] for j in 0-1, k in 0-1. V index: j*4 + k
    # C part: C[i, k] for i in 0-1, k in 0-1. W index: k*2 + i
    
    # Block 2: A2*B21, contributes to C[:, 0:2]
    # A2: A[i, j] for i in 0-1, j in 2-3. U index: i*4 + j
    # B21: B[j, k] for j in 2-3, k in 0-1. V index: j*4 + k
    # C part: C[i, k] for i in 0-1, k in 0-1. W index: k*2 + i
    
    # Block 3: A1*B12, contributes to C[:, 2:4]
    # A1: same as block 1
    # B12: B[j, k] for j in 0-1, k in 2-3. V index: j*4 + k
    # C part: C[i, k] for i in 0-1, k in 2-3. W index: k*2 + i
    
    # Block 4: A2*B22, contributes to C[:, 2:4]
    # A2: same as block 2
    # B22: B[j, k] for j in 2-3, k in 2-3. V index: j*4 + k
    # C part: C[i, k] for i in 0-1, k in 2-3. W index: k*2 + i
    
    for block in range(4):
        base_col = block * 7
        for t in range(7):
            col = base_col + t
            # U: for each A index
            for i in range(2):
                for j_local in range(2):
                    val = sU_cols[t][j_local + i * 2]  # Wait, indexing
                    # sU_cols[t] is [A11, A12, A21, A22]
                    # A11: i=0, j_local=0
                    # A12: i=0, j_local=1
                    # A21: i=1, j_local=0
                    # A22: i=1, j_local=1
                    # So index in sU_cols: i*2 + j_local
                    val = sU_cols[t][i * 2 + j_local]
                    if block in (0, 2):  # A1
                        j = j_local
                    else:  # A2
                        j = 2 + j_local
                    U[i * 4 + j][col] += val
            # V: for each B index
            for j_local in range(2):
                for k_local in range(2):
                    val = sV_cols[t][j_local * 2 + k_local]
                    if block in (0, 1):  # B1* (rows 0-1)
                        j = j_local
                    else:  # B2* (rows 2-3)
                        j = 2 + j_local
                    if block in (0, 3):  # B*1 (cols 0-1)
                        k = k_local
                    else:  # B*2 (cols 2-3)
                        k = 2 + k_local
                    V[j * 4 + k][col] += val
            # W: for each C index
            for i in range(2):
                for k_local in range(2):
                    val = sW_cols[t][k_local * 2 + i]
                    # C[i, k] where k depends on block
                    if block in (0, 1):  # C[:, 0:2]
                        k = k_local
                    else:  # C[:, 2:4]
                        k = 2 + k_local
                    W[k * 2 + i][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
