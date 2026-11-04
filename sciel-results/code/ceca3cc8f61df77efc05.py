import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to find a better decomposition using structured ansatz
    # Based on the idea of decomposing the 4x5x5 multiplication into smaller blocks
    # and using known low-rank decompositions for those blocks
    
    # The 4x5x5 multiplication can be seen as:
    # M_4,5,5 = M_4,4,4 ⊕ M_4,1,4 ⊕ M_4,4,1 ⊕ M_4,1,1
    # where M_4,4,4 is 4x4x4 (rank 16 with Laderman)
    # M_4,1,4 is 4x1x4 (rank 1)
    # M_4,4,1 is 4x4x1 (rank 1)
    # M_4,1,1 is 4x1x1 (rank 1)
    # Total: 16 + 1 + 1 + 1 = 19
    
    # However, this decomposition doesn't directly apply because the dimensions don't align perfectly.
    # Let me think about this differently.
    
    # Actually, for 4x5x5, we can use the following approach:
    # Split the 5x5 matrix into blocks:
    # B = [B_11 B_12; B_21 B_22] where B_11 is 4x4, B_12 is 4x1, B_21 is 1x4, B_22 is 1x1
    # A = [A_1; A_2] where A_1 is 4x4, A_2 is 4x1
    # C = A * B = [A_1*B_11 + A_2*B_21, A_1*B_12 + A_2*B_22]
    # C_1 is 4x4, C_2 is 4x1
    
    # So we need:
    # 1. 4x4x4 multiplication (A_1 * B_11) - rank 16 with Laderman
    # 2. 4x1x4 multiplication (A_2 * B_21) - rank 1 (outer product)
    # 3. 4x4x1 multiplication (A_1 * B_12) - rank 1 (outer product)  
    # 4. 4x1x1 multiplication (A_2 * B_22) - rank 1 (outer product)
    
    # Total rank: 16 + 1 + 1 + 1 = 19
    
    # Let me construct this decomposition
    rank = 19
    U = [[0] * rank for _ in range(n * m)]  # 4*5 = 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*5 = 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*4 = 20 rows
    
    # Laderman's 4x4x4 decomposition
    # This is a known rank-16 decomposition for 4x4 matrix multiplication
    # The coefficients are in {0, 1, -1}
    
    # Laderman's algorithm for 4x4 matrices
    # M_1 = (A_11 + A_22)(B_11 + B_22)
    # M_2 = (A_11 - A_22)(B_11 - B_22)
    # M_3 = (A_12 + A_21)(B_12 + B_21)
    # M_4 = (A_12 - A_21)(B_12 - B_21)
    # M_5 = (A_11 + A_21 + A_22)(B_11 + B_21)
    # M_6 = (A_11 + A_21)(B_11 + B_21 + B_22)
    # M_7 = (A_12 + A_22)(B_12 + B_22)
    # M_8 = (A_12 + A_22 + A_11)(B_12 + B_22)
    # M_9 = (A_11 + A_12 + A_22)(B_11 + B_12 + B_22)
    # M_10 = (A_11 + A_12)(B_11 + B_12 + B_21 + B_22)
    # M_11 = (A_21 + A_22)(B_11 + B_12 + B_21 + B_22)
    # M_12 = (A_11 + A_21 + A_12 + A_22)(B_11 + B_21)
    # M_13 = (A_11 + A_21)(B_12 + B_22)
    # M_14 = (A_12 + A_22)(B_11 + B_21)
    # M_15 = (A_11 + A_12 + A_21 + A_22)(B_11 + B_12 + B_21 + B_22)
    # M_16 = (A_11 + A_12 + A_21)(B_12 + B_22 + B_21)
    
    # Actually, let me use a simpler approach. I'll use the known Strassen-like decomposition
    # for 4x4 matrices, or just use the schoolbook for 4x4 (rank 16) and see if I can improve.
    
    # For now, let me just use the schoolbook decomposition for the 4x4 part (rank 16)
    # and rank-1 for the other parts.
    
    # Indexing:
    # A is 4x5: A[i,j] where i in 0..3, j in 0..4
    # B is 5x5: B[j,k] where j in 0..4, k in 0..4
    # C is 4x5: C[i,k] where i in 0..3, k in 0..4
    
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #         = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    # Let me split:
    # Part 1: sum_{j=0}^{3} A[i,j] * B[j,k] for i in 0..3, k in 0..4
    #         This is a 4x4x5 multiplication
    # Part 2: A[i,4] * B[4,k] for i in 0..3, k in 0..4
    #         This is a 4x1x5 multiplication (outer product)
    
    # For Part 1 (4x4x5):
    # C1[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # This can be further split:
    # C1[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for k in 0..3 + A[i,3] * B[3,4]... no
    # Actually, B is 4x5, so B[j,k] for j in 0..3, k in 0..4
    # Let me split k into 0..3 and 4:
    # C1[i,k] for k in 0..3: sum_{j=0}^{3} A[i,j] * B[j,k] -> 4x4x4 (rank 16)
    # C1[i,4]: sum_{j=0}^{3} A[i,j] * B[j,4] -> 4x4x1 (rank 1, outer product)
    
    # So total:
    # 4x4x4: rank 16
    # 4x4x1: rank 1
    # 4x1x5: rank 1 (outer product)
    # Total: 18
    
    # Let me construct this
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3)
    # Schoolbook: 16 terms
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Part 2: 4x4x1 (i in 0..3, j in 0..3, k=4)
    # C[i,4] += sum_{j=0}^{3} A[i,j] * B[j,4]
    # This is an outer product: A_4x4 * B_4x1
    # We can do this with 1 term if we use the structure, but actually it's rank 4
    # Wait, no. sum_{j=0}^{3} A[i,j] * B[j,4] is a dot product, not an outer product.
    # This is a 4x4x1 tensor, which has rank 4 (one term per i, with V depending on j)
    # Actually, T[i,j,4] = 1 for i in 0..3, j in 0..3
    # This tensor has rank 4 (it's the outer product of a 4-vector and a 4-vector, but transposed)
    # Let me use 4 terms:
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Part 3: 4x1x5 (i in 0..3, j=4, k in 0..4)
    # C[i,k] += A[i,4] * B[4,k]
    # This is an outer product: A_4x1 * B_1x5
    # T[i,4,k] = 1 for i in 0..3, k in 0..4
    # This has rank 1? No, it's an outer product of a 4-vector and a 5-vector, so rank 1
    # Wait, T[i,4,k] = 1 means for each (i,k), we have one term.
    # Actually, this is a rank-1 tensor: U[i,4] * V[4,k] * W[k,i]
    # But U depends on i, V depends on k, W depends on k,i.
    # Let me use 1 term:
    # U[i,4] = 1 for all i
    # V[4,k] = 1 for all k
    # W[k,i] = 1 for all k,i
    # Then the product is 1*1*1 = 1, which is correct.
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for k in range(5):
        V[4 * 5 + k][col] = 1
    for k in range(5):
        for i in range(4):
            W[k * 4 + i][col] = 1
    col += 1
    
    # Verify the decomposition
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if np.allclose(reconstructed, T, atol=1e-6):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
