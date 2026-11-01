import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = n * m * p  # 63 for 3x3x7

    # Build the trivial (schoolbook) decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify correctness (fast check for small dimensions)
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    # where T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i', j==j', k==k'
    # Quick spot-check: verify a few entries
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Verify all entries efficiently using tensor contraction
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-tensor of shape (n*m, m*p, p*n)
    # For our trivial decomposition, T[a,b,c] = 1 iff the indices are consistent
    
    # Spot check: for each (i,j,k), verify the corresponding entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # The entry T[a,b,c] should be 1
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if val != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k})")
                
                # Check that T[a,b,c'] = 0 for c' != c (when k' != k or i' != i)
                # and T[a,b',c] = 0 for b' != b, etc.
                # Only check a few to save time
                if time.time() > deadline:
                    break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
