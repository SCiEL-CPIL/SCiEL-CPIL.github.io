"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Construct a rank-18 decomposition for 2x3 x 3x6 -> 2x6
    # Strategy: use 18 rank-1 tensors
    
    # The key insight: for each j in {0,1,2}, we can use 6 rank-1 terms
    # to handle the contribution of A[:,j] * B[j,:] to C.
    # 
    # For fixed j, we need to compute:
    #   C[i,k] += A[i,j] * B[j,k] for i in {0,1}, k in {0..5}
    # 
    # We can use the following 6 terms for each j:
    #   For k in {0..5}:
    #     Term: U = e_{0j} + e_{1j} (sum of both A rows for column j)
    #           V = e_{jk} (B[j,k])
    #           W = e_{k0} + e_{k1} (sum of both C rows for column k)
    #     This gives (A[0,j]+A[1,j]) * B[j,k] contributing to C[0,k]+C[1,k]
    #     
    #   But we need C[0,k] and C[1,k] separately, not their sum.
    # 
    # Alternative: use 2 terms per k:
    #   Term 1: U = e_{0j} - e_{1j}, V = e_{jk}, W = e_{k0} - e_{k1}
    #   Term 2: U = e_{0j} + e_{1j}, V = e_{jk}, W = e_{k0} + e_{k1}
    #   From these: (Term1 + Term2)/2 = e_{0j}*e_{jk}*e_{k0} = A[0,j]*B[j,k] -> C[0,k]
    #   (Term2 - Term1)/2 = e_{1j}*e_{jk}*e_{k1} = A[1,j]*B[j,k] -> C[1,k]
    #   But we can't do division in the tensor decomposition; we need exact coefficients.
    # 
    # Actually, we CAN use coefficients. Let me use:
    #   Term 1: U = e_{0j}, V = e_{jk}, W = e_{k0}  -> A[0,j]*B[j,k] -> C[0,k]
    #   Term 2: U = e_{1j}, V = e_{jk}, W = e_{k1}  -> A[1,j]*B[j,k] -> C[1,k]
    # That's 2 terms per (j,k), total 36.
    # 
    # To get 18, we need to combine. The trick:
    #   For each j, use 6 terms where each term handles BOTH i=0 and i=1
    #   by using a 2-dimensional U and W.
    # 
    # But in tensor decomposition, each rank-1 term has scalar U, V, W vectors.
    # 
    # I think the minimum rank is actually 18, and here's a construction:
    # 
    # For each j in {0,1,2}:
    #   For each k in {0..5}:
    #     We need to contribute A[0,j]*B[j,k] to C[0,k] and
    #     A[1,j]*B[j,k] to C[1,k]
    # 
    # Use 2 terms per (j,k) but with shared V:
    #   Term A: U = [0..0, 1, 0..0] (A[0,j]), V = [0..0, 1, 0..0] (B[j,k]), W = [0..0, 1, 0..0] (C[0,k])
    #   Term B: U = [0..0, 0, 1, 0..0] (A[1,j]), V = [0..0, 1, 0..0] (B[j,k]), W = [0..0, 0, 1, 0..0] (C[1,k])
    # 
    # The V vector is the same for both terms! So we can "merge" them:
    #   Combined term: U = [0..0, 1, 1, 0..0] (A[0,j]+A[1,j]), V = B[j,k], W = [0..0, 1, 1, 0..0] (C[0,k]+C[1,k])
    #   This gives (A[0,j]+A[1,j])*B[j,k] = A[0,j]*B[j,k] + A[1,j]*B[j,k]
    #   But we need them separately in C[0,k] and C[1,k].
    # 
    # So merging doesn't work directly.
    # 
    # After more research, I'll implement a known rank-18 algorithm.
    # For 2x3 x 3x6, the rank is 18, and here's a valid construction:
    
    rank = 18
    U = [[0]*rank for _ in range(n*m)]  # 6x18
    V = [[0]*rank for _ in range(m*p)]  # 18x18
    W = [[0]*rank for _ in range(p*n)]  # 12x18
    
    # Construction using 18 terms:
    # Terms 0-5: for j=0, k=0..5
    # Terms 6-11: for j=1, k=0..5
    # Terms 12-17: for j=2, k=0..5
    
    # For each (j, k), use 2 terms:
    #   Term 2*(j*6+k): A[0,j]*B[j,k] -> C[0,k]
    #   Term 2*(j*6+k)+1: A[1,j]*B[j,k] -> C[1,k]
    # 
    # But that's 36 terms. To get 18, I need a different approach.
    
    # Let me use the following verified construction:
    # For each j, use 6 terms where the U and W vectors are 2-dimensional
    # in a combined sense.
    
    # Actually, I'll just implement the schoolbook (rank 36) for correctness,
    # but I know this isn't optimal. Let me try to find a rank-18 algorithm.
    
    # After careful thought, here's a rank-18 construction:
    # Use the fact that C = A*B can be written as:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # 
    # Define:
    #   s_j = A[0,j] + A[1,j] for j=0,1,2
    #   d_j = A[0,j] - A[1,j] for j=0,1,2
    # 
    # Then:
    #   A[0,j] = (s_j + d_j)/2
    #   A[1,j] = (s_j - d_j)/2
    # 
    # So:
    #   C[0,k] = sum_j (s_j + d_j)/2 * B[j,k] = (1/2) * sum_j s_j*B[j,k] + (1/2) * sum_j d_j*B[j,k]
    #   C[1,k] = sum_j (s_j - d_j)/2 * B[j,k] = (1/2) * sum_j s_j*B[j,k] - (1/2) * sum_j d_j*B[j,k]
    # 
    # Let S_k = sum_j s_j * B[j,k] and D_k = sum_j d_j * B[j,k]
    # Then C[0,k] = (S_k + D_k)/2 and C[1,k] = (S_k - D_k)/2
    # 
    # Now, S_k is a linear combination of B[j,k] with coefficients s_j.
    # Similarly for D_k.
    # 
    # To compute S_k for all k, we need 3*6 = 18 multiplications (s_j * B[j,k]).
    # Similarly for D_k: 18 multiplications.
    # Total: 36 multiplications, same as before.
    # 
    # But we can share: S_k and D_k both use B[j,k], so we can compute
    # s_j*B[j,k] and d_j*B[j,k] together.
    # 
    # For each (j,k), compute:
    #   x = s_j * B[j,k]
    #   y = d_j * B[j,k]
    # That's 2 multiplications per (j,k), total 36.
    # 
    # I'm stuck. Let me just implement rank 36 for now.
    
    rank = n * m * p  # 36
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
# EVOLVE-BLOCK-END
