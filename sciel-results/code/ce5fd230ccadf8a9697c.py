import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 3, 3, 7
    
    # Helper to create schoolbook decomposition for n x m x p_block
    # offset_k is the starting k index in the global p dimension
    def schoolbook_block(n, m, p_block, offset_k):
        rank = n * m * p_block
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    k_global = k + offset_k
                    U[i * m + j, col] = 1.0
                    V[j * p + k_global, col] = 1.0
                    W[k_global * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Block 1: k = 0, 1, 2
    U1, V1, W1, r1 = schoolbook_block(n, m, 3, 0)
    
    # Block 2: k = 3, 4, 5
    U2, V2, W2, r2 = schoolbook_block(n, m, 3, 3)
    
    # Block 3: k = 6 (3x3x1)
    # Rank 3 decomposition for 3x3x1
    # T[i, j, 6] = 1 if i is the output row.
    # We need C_i = sum_j A_ij B_j6.
    # Use 3 terms, one for each j?
    # Term r (j=r):
    # U[i, r, r] = 1 if i is any? No.
    # For fixed j, we need to multiply A_ij by B_j6.
    # U[i, j, r] should be 1 for all i? No, U depends on i, j.
    # Let's use rank 3.
    # r=0: j=0. U[i, 0, 0]=1. V[0, 6, 0]=1. W[6, i, 0]=1.
    # r=1: j=1. U[i, 1, 1]=1. V[1, 6, 1]=1. W[6, i, 1]=1.
    # r=2: j=2. U[i, 2, 2]=1. V[2, 6, 2]=1. W[6, i, 2]=1.
    # Sum for C_i: sum_r (sum_{a,b} U[a,b,r] A_ab) (sum_c V[c,r] B_c) W[6,i,r]
    # For r=0: (sum_i A_i0) * B_06 * 1.
    # This gives sum_i A_i0 * B_06.
    # We want A_i0 * B_06.
    # So U must be 1 only for the specific i?
    # If U[i, 0, 0] = 1 for all i, then sum_i A_i0 is the sum of the column.
    # We need to isolate A_i0.
    # So U[i, 0, 0] should be 1 if i is the target row?
    # But W[6, i, 0] selects the row i.
    # If we set U[i, 0, 0] = 1 for all i, then the term is (Sum Col 0 of A) * B_06.
    # This is wrong.
    # We need U to pick out A_i0.
    # So U[i, 0, 0] = 1, and U[i', 0, 0] = 0 for i' != i?
    # But U is a matrix of size 9x3.
    # If we use 3 terms, one for each j, we can't vary i in U independently for each term without increasing rank.
    # Actually, the rank of 3x3x1 is 3.
    # The decomposition is:
    # r=0: U[:, 0, 0] = e_0 (1 at i=0, j=0). V[0, 6, 0] = 1. W[6, 0, 0] = 1.
    # This gives A_00 B_06 for C_0.
    # We need all i.
    # So we need 9 terms?
    # Rank of 3x3x1 is 9?
    # Yes, schoolbook for 3x3x1 is rank 9.
    # So total rank is 27 + 27 + 9 = 63.
    # Back to square one.

    # Wait, is Rank(3x3x1) really 9?
    # C = A v.
    # This is a linear map A -> C.
    # The tensor is T[i, j, i] = 1?
    # No, T[i, j, k] with k=i.
    # This is the identity tensor.
    # The rank of the identity tensor of size n x n is n.
    # But here the middle index j is summed.
    # It is the tensor for matrix-vector multiplication.
    # The rank is n * m?
    # Yes, it is the schoolbook rank.
    
    # So the block decomposition does not help if the blocks are not independent in A.
    # But they ARE independent in the tensor structure?
    # No, the U matrices are shared in the global tensor?
    # No, we stack them.
    # If Rank(3x3x1) = 9, then total is 63.
    
    # I will return the schoolbook 63.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
