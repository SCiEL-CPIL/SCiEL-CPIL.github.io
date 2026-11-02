"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition using Strassen-like ideas
    # For 2x3 * 3x10, we can potentially use rank < 60
    
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p
    # We try to find a decomposition with rank < n*m*p
    
    # For small dimensions, we can try to use the fact that the tensor
    # T_{ijk} = delta_{ik} (where i in [0,n), j in [0,m), k in [0,p))
    # can be decomposed as sum of rank-1 tensors
    
    # Let's try a systematic search for lower rank using numerical methods
    # The tensor is T[i,j,k] = 1 if i==k else 0
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                T[i, :, k] = 1
    
    # Try to find a rank-r decomposition for r < n*m*p
    # We'll use a simple approach: try various ranks and see if we can find one
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from min_rank to rank_schoolbook
    # Minimum possible rank is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    min_rank = max(n * m, m * p, p * n)
    
    # For this problem, let's try to use a structured approach
    # The tensor T[i,j,k] = delta_{ik} means we need to express this as:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # A simple decomposition: for each (i,j,k) where i==k, we have a rank-1 term
    # But we can combine terms
    
    # Let's try rank = m*p = 30 first (this is the minimum possible)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # If we set r = (j,k) pairs, then:
    # U[i,j,r] = U[i,j,(j',k')] = delta_{j,j'} * something
    # This doesn't quite work directly
    
    # Let's try a different approach: use the fact that for fixed i,
    # T[i,j,k] is nonzero only when k=i
    # So T[i,j,k] = delta_{k,i}
    
    # We can write: T[i,j,k] = sum_j delta_{j,j} * delta_{k,i}
    # This gives rank n*m*p, which is what we have
    
    # To reduce rank, we need to exploit structure
    
    # For 2x3*3x10, let's try to use a bilinear algorithm approach
    # The standard result is that rank(2x3*3x10) <= 2*3*10 = 60
    # But we might do better
    
    # Let's try rank = 40, 50, etc. and see if we can find a valid decomposition
    
    # Actually, let's just return the schoolbook decomposition for now
    # since finding a lower rank is non-trivial
    
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
