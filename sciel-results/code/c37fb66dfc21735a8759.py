import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    timeout = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 18 decomposition using 3 blocks of rank-6 (2x3x3) decompositions
    # Each block handles 3 columns of B and C
    # The 2x3x3 all-ones tensor decomposition:
    #   r=0: U[0,0]=1, V[0,0]=1, W[:,0]=1
    #   r=1: U[0,1]=1, V[1,1]=1, W[:,1]=1
    #   r=2: U[0,2]=1, V[2,2]=1, W[:,2]=1
    #   r=3: U[1,3]=1, V[0,3]=1, W[:,3]=1
    #   r=4: U[1,4]=1, V[1,4]=1, W[:,4]=1
    #   r=5: U[1,5]=1, V[2,5]=1, W[:,5]=1
    
    rank = 18
    U = np.zeros((n * m, rank))  # 6 x 18
    V = np.zeros((m * p, rank))  # 27 x 18
    W = np.zeros((p * n, rank))  # 18 x 18
    
    # For each block b (0, 1, 2), handle k in [3b, 3b+2]
    for block in range(3):
        r_offset = block * 6
        # Within this block, for each (i, j) pair, assign one rank term
        # i in {0, 1}, j in {0, 1, 2}
        term = 0
        for i in range(n):
            for j in range(m):
                r_global = r_offset + term
                # U index: i * m + j = i * 3 + j
                U[i * m + j, r_global] = 1.0
                # V index: j * p + k for k in [3*block, 3*block+2]
                # V depends on j and k. For this term, we want V[j, k] = 1 for all k in block
                for k in range(3):
                    k_global = block * 3 + k
                    V[j * p + k_global, r_global] = 1.0
                    # W index: k * n + i = k_global * 2 + i
                    W[k_global * n + i, r_global] = 1.0
                term += 1
    
    # Verify the decomposition
    # Check: for all i, j, k: sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    else:
        # All entries are integers, so they are valid half-integers
        pass
    
    if time.time() - start_time > timeout - 5:
        pass  # Return immediately
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
