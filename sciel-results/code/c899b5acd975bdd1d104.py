"""
Matrix multiplication rank minimization task.
Dimensions: 2x3 * 3x10 = 2x10
We use the standard schoolbook algorithm which has rank n*m*p.
For n=2, m=3, p=10, rank = 60.
"""

import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Enforce a strict time limit to avoid timeouts, although this computation is trivial.
    start_time = time.time()
    deadline = 100.0 # 100 seconds out of 120 allowed.

    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T = sum_{i,j,k} (A_{i,j}) (B_{j,k}) (C_{i,k})
    # Rank is n*m*p.
    
    rank = n * m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Note: The problem statement says W has shape (p*n, rank) and follows trace convention W[k*n+i, r].
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (as per problem statement W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
                # Check time limit, though this loop runs < 1ms
                if time.time() > deadline:
                    # Return current partial result? No, must return valid result.
                    # Since it's so fast, this is just a safety guard.
                    break
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
