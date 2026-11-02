import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 108  # safety margin
    
    n, m, p = 3, 5, 7
    
    # Trivial rank-105 decomposition as fallback
    trivial_rank = n * m * p
    U_trivial = [[0] * trivial_rank for _ in range(n * m)]
    V_trivial = [[0] * trivial_rank for _ in range(m * p)]
    W_trivial = [[0] * trivial_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = trivial_rank
    
    # Block-diagonal decomposition: treat 3x5 multiplication as three 1x5 row-vector multiplications
    # Each 1x5x7 tensor needs rank 5 (identity basis)
    # Total rank = 3 * 5 = 15
    # 
    # For row i (i=0,1,2), the 1x5 matrix A_i = [a_0, a_1, a_2, a_3, a_4]
    # Multiplied by 5x7 matrix B gives row vector C_i of length 7
    # C_i[k] = sum_{j=0}^{4} A_i[j] * B[j][k]
    #
    # Tensor decomposition: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # For the block-diagonal approach:
    # For each row i, use 5 rank-1 terms (one for each j in 0..4)
    # Term t corresponds to (i, j) pair
    # U[i*m+j, t] = 1 if t encodes (i,j), else 0
    # V[j*p+k, t] = 1 if t encodes (i,j) and k is the output index... 
    #
    # Wait, let me think more carefully.
    #
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # Block-diagonal idea: For each row i, we have a 1x5x7 sub-tensor
    # The sub-tensor for row i has entries T[i,j,k] = 1 for j in 0..4, k in 0..6
    # This is a rank-5 tensor: T[i,j,k] = sum_{j'=0}^{4} delta_{j,j'} * delta_{k,j'}... no that's not right
    #
    # Actually, the 1x5x7 tensor with all ones has rank 5 (or 7, whichever is smaller)
    # T[i,j,k] = sum_{t=0}^{4} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # For the all-ones 1x5x7 tensor, one decomposition is:
    # Use 5 terms, one for each column of the 5x7 matrix
    # Term t (t=0..4): 
    #   U[i*m+j, t] = 1 for all j (since we're summing over j)
    #   V[j*p+k, t] = 1 if j == t, else 0
    #   W[k*n+i, t] = 1 for all k
    #
    # Check: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # = sum_{t=0}^{4} 1 * delta_{j,t} * 1 = 1 if j in 0..4, which is always true
    # 
    # So for each row i, we need 5 terms. Total = 3 * 5 = 15 terms.
    #
    # Let's construct this:
    # Term index t_global = i * 5 + j_local where j_local in 0..4
    # U[i*m+j, t_global] = 1 if t_global // 5 == i, else 0
    # V[j*p+k, t_global] = 1 if j == t_global % 5, else 0
    # W[k*n+i, t_global] = 1 if t_global // 5 == i, else 0
    
    block_rank = n * m  # 3 * 5 = 15
    U_block = [[0] * block_rank for _ in range(n * m)]
    V_block = [[0] * block_rank for _ in range(m * p)]
    W_block = [[0] * block_rank for _ in range(p * n)]
    
    for i in range(n):
        for j_local in range(m):  # j_local is the column index in the 1x5 row
            t_global = i * m + j_local
            # U: for row i, all j entries get 1 for this term
            for j in range(m):
                U_block[i * m + j][t_global] = 1
            # V: for column j_local, all k entries get 1 if j == j_local
            for k in range(p):
                V_block[j_local * p + k][t_global] = 1
            # W: for row i, all k entries get 1
            for k in range(p):
                W_block[k * n + i][t_global] = 1
    
    # Verify the block decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    if verify(U_block, V_block, W_block, n, m, p):
        best_result = {"U": U_block, "V": V_block, "W": W_block}
        best_rank = block_rank
    
    return best_result
