import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 blocks of 2x2x2 (rank 7 each) for k=0,1 and k=2,3
    # 1 block of 2x4x1 (rank 8) for k=4
    # Total rank = 7*2 + 7*2 + 8 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off, col_idx):
        """
        Adds a 2x2x2 Strassen block at offsets i_off, j_off, k_off.
        i_off: offset for row index of A (and C)
        j_off: offset for col index of A (and row index of B)
        k_off: offset for col index of B (and C)
        """
        # M1 = (A00 + A11)(B00 + B11)
        U[i_off + 0*m + j_off][col_idx] += 1
        U[i_off + 1*m + j_off + 1][col_idx] += 1
        V[j_off + 0*p + k_off][col_idx] += 1
        V[j_off + 1*p + k_off + 1][col_idx] += 1
        W[k_off + 0*n + i_off][col_idx] += 1
        W[k_off + 1*n + i_off + 1][col_idx] += 1
        
        # M2 = (A10 + A11)B00
        U[i_off + 1*m + j_off][col_idx] += 1
        U[i_off + 1*m + j_off + 1][col_idx] += 1
        V[j_off + 0*p + k_off][col_idx] += 1
        W[k_off + 0*n + i_off + 1][col_idx] += 1
        
        # M3 = A00(B10 - B00)
        U[i_off + 0*m + j_off][col_idx] += 1
        V[j_off + 1*p + k_off][col_idx] += 1
        V[j_off + 0*p + k_off][col_idx] += -1
        W[k_off + 1*n + i_off][col_idx] += 1
        
        # M4 = A11(B01 - B11)
        U[i_off + 1*m + j_off + 1][col_idx] += 1
        V[j_off + 0*p + k_off + 1][col_idx] += 1
        V[j_off + 1*p + k_off + 1][col_idx] += -1
        W[k_off + 0*n + i_off][col_idx] += 1
        W[k_off + 0*n + i_off + 1][col_idx] += 1
        
        # M5 = (A00 + A01)B11
        U[i_off + 0*m + j_off][col_idx] += 1
        U[i_off + 0*m + j_off + 1][col_idx] += 1
        V[j_off + 1*p + k_off + 1][col_idx] += 1
        W[k_off + 1*n + i_off][col_idx] += 1
        W[k_off + 0*n + i_off][col_idx] += -1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[i_off + 1*m + j_off][col_idx] += 1
        U[i_off + 0*m + j_off][col_idx] += -1
        V[j_off + 0*p + k_off][col_idx] += 1
        V[j_off + 1*p + k_off][col_idx] += 1
        W[k_off + 1*n + i_off + 1][col_idx] += 1
        
        # M7 = (A01 - A11)(B01 + B11)
        U[i_off + 0*m + j_off + 1][col_idx] += 1
        U[i_off + 1*m + j_off + 1][col_idx] += -1
        V[j_off + 0*p + k_off + 1][col_idx] += 1
        V[j_off + 1*p + k_off + 1][col_idx] += 1
        W[k_off + 0*n + i_off][col_idx] += 1

    # Block 1: k=0,1
    # A_L (j=0,1) * B_L (j=0,1)
    add_strassen_block(0, 0, 0, col); col += 7
    # A_R (j=2,3) * B_R (j=2,3)
    add_strassen_block(0, 2, 0, col); col += 7
    
    # Block 2: k=2,3
    # A_L (j=0,1) * B_L (j=0,1)
    add_strassen_block(0, 0, 2, col); col += 7
    # A_R (j=2,3) * B_R (j=2,3)
    add_strassen_block(0, 2, 2, col); col += 7
    
    # Block 3: k=4
    # Schoolbook for 2x4x1
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
