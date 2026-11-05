import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # The matrix multiplication tensor for (n x m) * (m x p) -> (p x n)
    # has entries T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # in the standard "all-ones" formulation where:
    #   (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor decomposition requires:
    #   T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # The naive (schoolbook) decomposition uses rank = n*m*p = 63.
    # We attempt to find a lower-rank decomposition using alternating least squares
    # or a structured approach. For now, we use the schoolbook as baseline
    # and try a few structured low-rank constructions.

    # --- Attempt 1: Try a structured rank-21 decomposition ---
    # For 3x3x7, we can think of the 7 columns as 3+3+1.
    # The 3x3x3 part has known rank at most 21 (schoolbook) or potentially lower.
    # But for 3x3x7, let's try a direct construction.
    #
    # Key insight: The tensor T[i,j,k] = 1 for all i,j,k is actually the all-ones
    # tensor, which has rank 1! But that's not the matrix multiplication tensor.
    #
    # Wait - let me re-read the problem. The matrix multiplication tensor is:
    #   MM_{n,m,p}[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the standard formulation, the tensor has entries that are 1 for the
    # "diagonal" pattern. Actually, for matrix multiplication:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k],
    # which is 1 for all i,j,k. So T is indeed the all-ones tensor of shape (n,m,p).
    #
    # The all-ones tensor of shape (n,m,p) has rank 1! Because:
    #   T[i,j,k] = 1 = U[i,j]*V[j,k]*W[k,i] where U=V=W=1 (all ones)
    #
    # But wait, that's not right either. The rank-1 decomposition requires:
    #   T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For rank 1: T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    # If U, V, W are all 1s, then T[i,j,k] = 1*1*1 = 1. Yes!
    #
    # So the all-ones tensor has rank 1. But is this the correct tensor?
    # Let me verify with the schoolbook: in the schoolbook, each term
    # U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1 only for the specific
    # (i,j,k) that corresponds to that column. So the sum over all columns gives
    # T[i,j,k] = 1 for each (i,j,k). This is indeed the all-ones tensor.
    #
    # Therefore, the matrix multiplication tensor for any dimensions is the
    # all-ones tensor, which has rank 1!

    # Let's verify: rank-1 decomposition
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))

    # Verify
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    if np.allclose(T_reconstructed, 1.0, atol=1e-9):
        # Rank 1 works!
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank-63
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
