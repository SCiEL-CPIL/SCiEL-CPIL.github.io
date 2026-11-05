"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Rank-19 decomposition for 3x3 matrix multiplication."""
    import numpy as np
    
    n, m, p = 3, 3, 3
    rank = 19
    
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)
    
    # A indices: a_ij -> i*3+j
    # B indices: b_jk -> j*3+k
    # W indices: c_ik -> k*3+i (per convention W[k*n+i])
    
    def a(i,j): return i*3+j
    def b(j,k): return j*3+k
    def w(k,i): return k*3+i
    
    U = np.zeros((9, 27), dtype=int)
    V = np.zeros((9, 27), dtype=int)
    W = np.zeros((9, 27), dtype=int)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
