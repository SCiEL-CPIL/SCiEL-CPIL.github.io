"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # The tensor for multiplying (n x m) by (m x p):
    # T[I, J, K] = 1 if I = i*m + j, J = j*p + k, K = k*n + i for some (i,j,k)
    # Otherwise T[I, J, K] = 0
    #
    # For n=2, m=2, p=15:
    # I is in {0,1,2,3}, J is in {0,...,29}, K is in {0,...,29}
    #
    # Valid positions:
    # i=0, j=0: I=0, J=15k, K=2k       for k=0..14
    # i=1, j=0: I=2, J=15k, K=2k+1     for k=0..14
    # i=0, j=1: I=1, J=15k+1, K=2k     for k=0..14
    # i=1, j=1: I=3, J=15k+1, K=2k+1   for k=0..14
    #
    # Total 60 valid positions out of 4*30*30 = 3600.
    #
    # Lower bound on rank: max(n*m, m*p, p*n) = max(4, 30, 30) = 30.
    #
    # Key insight: For each k, we have 4 valid positions:
    # (0, 15k, 2k), (2, 15k, 2k+1), (1, 15k+1, 2k), (3, 15k+1, 2k+1)
    #
    # We can construct a rank-30 decomposition using 2 terms per k.
    # Term 2k:   U[0,2k]=1, U[1,2k]=1,  V[15k,2k]=1, V[15k+1,2k]=1,  W[2k,2k]=1, W[2k+1,2k]=1
    #            This gives: T[0,15k,2k] += 1*1*1 = 1 (i=0,j=0)
    #                       T[1,15k,2k] += 1*1*1 = 1 (but this should be 0! i=0,j=1 has J=15k+1)
    # Wait, let me reconsider.
    #
    # Actually, for term 2k:
    # U[0, 2k] = 1, U[1, 2k] = 0, U[2, 2k] = 0, U[3, 2k] = 0
    # V[15k, 2k] = 1, others 0
    # W[2k, 2k] = 1, others 0
    # This gives T[0, 15k, 2k] = 1. Good for (i=0,j=0).
    #
    # But we need to also cover (i=0,j=1): T[1, 15k+1, 2k] = 1
    # And (i=1,j=0): T[2, 15k, 2k+1] = 1
    # And (i=1,j=1): T[3, 15k+1, 2k+1] = 1
    #
    # With only 2 terms per k, we can't do it with simple 0/1 entries.
    # Let's try a different approach.
    #
    # Actually, let's use the structure more carefully.
    # For fixed k, we need to satisfy:
    # sum_r U[0,r]V[15k,r]W[2k,r] = 1  (i=0,j=0)
    # sum_r U[1,r]V[15k+1,r]W[2k,r] = 1  (i=0,j=1)
    # sum_r U[2,r]V[15k,r]W[2k+1,r] = 1  (i=1,j=0)
    # sum_r U[3,r]V[15k+1,r]W[2k+1,r] = 1  (i=1,j=1)
    # And all other combinations must give 0.
    #
    # If we use exactly 2 terms per k (terms 2k and 2k+1), and zero all other terms
    # for the relevant rows/columns, we need:
    #
    # For term 2k: U[0,a]V[15k,b]W[2k,c] + U[0,d]V[15k,e]W[2k,f] = 1
    # But if only term 2k contributes to V[15k,:] and W[2k,:], then:
    # U[0,2k] * V[15k,2k] * W[2k,2k] = 1
    #
    # Let's try:
    # Term 2k:   U[0,2k]=1, U[2,2k]=1,  V[15k,2k]=1,  W[2k,2k]=1, W[2k+1,2k]=1
    #   Contributes: T[0,15k,2k] += 1*1*1 = 1  (good, i=0,j=0)
    #                T[0,15k,2k+1] += 1*1*1 = 1 (BAD, should be 0)
    #                T[2,15k,2k] += 1*1*1 = 1 (BAD, should be 0)
    #                T[2,15k,2k+1] += 1*1*1 = 1  (good, i=1,j=0)
    #
    # This doesn't work. Let me try a cleaner construction.
    #
    # Schoolbook rank is 60. Can we do 30?
    # The lower bound is 30. Let's try to achieve rank 30.
    #
    # Idea: Use the fact that the problem decomposes by k.
    # For each k, we have a 2x2 sub-problem (i,j) -> (I,K) with fixed J pattern.
    # The 2x2 matrix multiplication (n=2, m=2, p=1) has rank at most 4, 
    # but actually the minimal rank for 2x2 x 2x1 is 4 (it's the same as 
    # computing 2 dot products of length 2).
    # Wait, 2x2 times 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is 4 bilinear terms, and they're all independent, so rank is 4.
    # So for each k, we need rank 4, giving total rank 4*15 = 60.
    #
    # Hmm, so 30 might not be achievable. The lower bound of 30 is not tight here.
    # The actual minimal rank is likely 60 (schoolbook).
    #
    # But wait - can we share terms across different k values?
    # The J index is j*p + k, so different k values give different J columns (for fixed j).
    # Similarly, K = k*n + i, so different k give different K rows.
    # So terms for different k affect different (J,K) positions.
    # Therefore, we cannot share terms across k values.
    #
    # So the minimal rank is 4 * p = 60, which is the schoolbook rank.
    #
    # Let's just return the schoolbook decomposition.

    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
