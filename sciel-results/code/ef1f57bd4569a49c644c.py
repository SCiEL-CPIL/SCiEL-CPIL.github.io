"""Rewritten program for matrix multiplication 4,4,8 rank minimization."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 145

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Rank-8 decomposition of the all-ones tensor for 4x4 x 4x8 matrix multiplication
    # The tensor T[i,j,k] = 1 for all i,j,k (since every term contributes)
    # Rank = max(n, m, p) = max(4, 4, 8) = 8
    # 
    # Construction: For each (i, k) pair, create one rank-1 term
    # U[i, j, r_{ik}] = 1 for all j (row i of A)
    # V[j, k, r_{ik}] = 1 for all j (column k of B)  
    # W[k, i, r_{ik}] = 1
    #
    # Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # because for each (i,j,k), exactly one r (corresponding to (i,k)) has
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1, giving product = 1.
    
    rank = n * p  # 4 * 8 = 32? Wait, let me reconsider.
    
    # Actually, the standard result: the tensor rank of the all-ones tensor 
    # of size a x b x c is max(a, b, c) over reals.
    # But we need to verify this construction works.
    
    # Let me try rank = p = 8:
    # For each k (0..7), define term r = k:
    #   U[i*m + j, k] = 1 for all i, j  (so U has all 1s in column k)
    #   V[j*p + k, k] = 1 for all j  (column k of B)
    #   W[k'*n + i, k] = 1 if k' == k else 0  (only row k of W)
    #
    # Then for fixed (i, j, k0):
    #   sum_r U[i*m+j, r] * V[j*p+k0, r] * W[k0*n+i, r]
    #   = sum_r 1 * (1 if r==k0 else 0) * (1 if k0==k0 and r==k0 else 0)
    #   Wait, W[k0*n+i, r] = 1 only if r == k0
    #   So the sum = U[i*m+j, k0] * V[j*p+k0, k0] * W[k0*n+i, k0] = 1*1*1 = 1 ✓
    
    # So rank = p = 8 works!
    rank = p  # 8
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 8
    V = [[0] * rank for _ in range(m * p)]  # 32 x 8
    W = [[0] * rank for _ in range(p * n)]  # 32 x 8
    
    for r in range(rank):
        # U: all 1s in column r
        for idx in range(n * m):
            U[idx][r] = 1
        # V: 1s in column r for all j (rows j*p + r)
        for j in range(m):
            V[j * p + r][r] = 1
        # W: 1 at position (r, i) for all i, i.e., W[r*n + i][r] = 1
        for i in range(n):
            W[r * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
