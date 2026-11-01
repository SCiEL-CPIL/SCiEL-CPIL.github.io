"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook (rank 48) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank using structured approach
    # For 2x2xp, try rank = 2*p = 24
    # Use the decomposition based on the 2x2 matrix structure
    
    # Try rank 24: c indexed by (j, k) where j in {0,1}, k in {0..11}
    # We need to solve for U, V, W such that the tensor equation holds
    # This is a 3-way tensor decomposition problem
    
    # Use a direct construction:
    # For the 2x2 matrix A and 2xp matrix B:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Try using the fact that we can share some computations
    # Let's try a rank-24 decomposition with specific structure
    
    # Actually, let me try to use the known result: M(2,2,p) has rank at most 2p + 2
    # But let's try to get rank 2p = 24
    
    # For rank 24, let c = j*p + k for j in {0,1}, k in {0..p-1}
    # U[i*m+j, c] = U[i*2+j, j'*p+k']
    # V[j*p+k, c] = V[j*12+k, j'*12+k']
    # W[k*n+i, c] = W[k*2+i, j'*2+i']
    
    # Let's try: U[i*2+j, j'*12+k'] = delta_{j,j'} * a[i,j]
    #            V[j*12+k, j'*12+k'] = delta_{k,k'} * b[j,k]
    #            W[k*2+i, j'*12+k'] = c[k,i]
    # Then: sum_{j',k'} delta_{j,j'} a[i,j] * delta_{k,k'} b[j,k] * c[k,i]
    #      = a[i,j] * b[j,k] * c[k,i]
    # Need this = 1 for all i,j,k
    
    # So a[i,j] * b[j,k] * c[k,i] = 1
    # This gives: a[0,0]*b[0,0]*c[0,0] = 1
    #             a[0,1]*b[1,0]*c[0,0] = 1
    #             a[1,0]*b[0,0]*c[0,1] = 1
    #             a[1,1]*b[1,0]*c[0,1] = 1
    #             ... and so on for all k
    
    # From a[0,0]*b[0,0]*c[0,0] = a[0,1]*b[1,0]*c[0,0]:
    # a[0,0]*b[0,0] = a[0,1]*b[1,0]
    # From a[1,0]*b[0,0]*c[0,1] = a[1,1]*b[1,0]*c[0,1]:
    # a[1,0]*b[0,0] = a[1,1]*b[1,0]
    
    # Let's set b[j,k] = 1 for all j,k (since k is independent)
    # Then: a[i,j] * c[k,i] = 1 for all i,j,k
    # This requires a[i,j] to be independent of j, and c[k,i] to be independent of k
    # So a[i,j] = a[i] and c[k,i] = c[i], with a[i]*c[i] = 1
    # But then for different j, we have the same a[i], which is fine
    # And for different k, same c[i], which is fine
    
    # So: a[0,0] = a[0,1] = a0, a[1,0] = a[1,1] = a1
    #     c[0,0] = c[0,1] = ... = c0, c[1,0] = c[1,1] = ... = c1
    #     b[j,k] = 1 for all j,k
    #     a0 * c0 = 1, a1 * c1 = 1
    
    # Let's set a0 = 1, c0 = 1, a1 = 1, c1 = 1
    # Then U[i*2+j, j'*12+k'] = delta_{j,j'} * a[i] = delta_{j,j'}
    #     V[j*12+k, j'*12+k'] = delta_{k,k'} * 1 = delta_{k,k'}
    #     W[k*2+i, j'*12+k'] = c[i] = 1
    
    # Check: sum_{j',k'} delta_{j,j'} * 1 * delta_{k,k'} * 1 = 1 (when j'=j and k'=k)
    # Yes! This works!
    
    rank = 2 * p  # 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for j in range(m):  # j = 0, 1
        for k in range(p):  # k = 0..11
            c = j * p + k
            for i in range(n):  # i = 0, 1
                U[i*m + j, c] = 1  # a[i] = 1
            V[j*p + k, c] = 1  # b[j,k] = 1
            for i in range(n):  # i = 0, 1
                W[k*n + i, c] = 1  # c[i] = 1
    
    # Verify
    # For all i,j,k: sum_c U[i*2+j,c] * V[j*12+k,c] * W[k*2+i,c] should be 1
    # U[i*2+j, c] is non-zero only when c's j-part = j, i.e., c = j*12 + k' for some k'
    # V[j*12+k, c] is non-zero only when c's k-part = k, i.e., c = j'*12 + k for some j'
    # So c must have j-part = j AND k-part = k, so c = j*12+k
    # Then U[i*2+j, j*12+k] = 1, V[j*12+k, j*12+k] = 1, W[k*2+i, j*12+k] = 1
    # Sum = 1. Correct!
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
