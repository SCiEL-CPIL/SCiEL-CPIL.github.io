"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def check_time():
        return time.time() > deadline
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        if not U or not V or not W:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False
        rank = U_arr.shape[1]
        if rank == 0 or V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Rank-50 decomposition using Strassen on 2x2 blocks with (2+1) inner split
    # Split inner dim m=3 into j=(0,1) and j=2
    # Part A: A1(2x2) * B1(2x9) -> use Strassen-based rank-48
    # Part B: A2(2x1) * B2(1x9) -> rank 2
    # Total: 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50
    
    # Part B: j=2, rank 2 (terms 48, 49)
    # C[i,k] += A[i,2] * B[2,k]
    # Term 48: i=0 contribution
    # Term 49: i=1 contribution
    # For term 48: U[0*3+2]=1, V[2*9+k]=1 for all k, W[k*2+0]=1 for all k
    # For term 49: U[1*3+2]=1, V[2*9+k]=1 for all k, W[k*2+1]=1 for all k
    for k in range(p):
        U[2][48] = 1  # i=0, j=2
        V[2 * p + k][48] = 1  # j=2, k
        W[k * n + 0][48] = 1  # k, i=0
        U[5][49] = 1  # i=1, j=2
        V[2 * p + k][49] = 1  # j=2, k
        W[k * n + 1][49] = 1  # k, i=1
    
    # Part A: j in {0,1}, 2x2 x 9, rank 48 (terms 0-47)
    # Use Strassen's 7 multiplications applied to each of the 9 columns,
    # but with sharing: 7*9 = 63 is too many. Instead use a different approach.
    # 
    # Actually, for 2x2 x 9: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Schoolbook: 4*9 = 36 terms. We need 48 terms for rank 50 total.
    # That means Part A uses 48 terms, which is MORE than schoolbook (36).
    # This doesn't make sense. Let me reconsider.
    #
    # If Part B uses rank 2, Part A must use rank 48 for total 50.
    # But Part A schoolbook is 36. So Part A rank 48 > 36 is impossible to improve.
    #
    # The rank-50 must come from a different decomposition strategy.
    # Let me just use schoolbook for now (rank 54) and verify it.
    
    # Fallback: schoolbook rank 54
    rank_sb = n * m * p  # 54
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
    
    if verify(U_sb, V_sb, W_sb, n, m, p):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Should not reach here
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
