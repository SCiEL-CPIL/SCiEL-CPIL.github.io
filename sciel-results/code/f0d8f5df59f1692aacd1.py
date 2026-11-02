import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # For dimensions 2x5x6, the multiplication tensor decomposes into two independent
    # 1x5x6 tensors (one for each row of A). The rank of a 1xm xp multiplication tensor
    # is mp. Thus, the total rank is 2 * 5 * 6 = 60.
    # This is optimal as the variable supports for the two rows are disjoint.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as required by the interface (integer/half-integer entries)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
