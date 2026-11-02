import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix. The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition as baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists of floats as required by the interface
    U_out = [[float(x) for x in U[i]] for i in range(n * m)]
    V_out = [[float(x) for x in V[i]] for i in range(m * p)]
    W_out = [[float(x) for x in W[i]] for i in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
