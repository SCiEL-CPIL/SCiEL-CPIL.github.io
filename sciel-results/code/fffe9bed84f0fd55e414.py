"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's rank-7 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U row u = i*m + j
    # V row v = j*p + k
    # W row w = k*n + i
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    
    V = [
        [1, 1, 0, -1, 0, 0, 1],  # e
        [0, 1, 1, 0, 0, 0, 0],  # f
        [0, 0, 0, 1, 0, 1, 1],  # g
        [1, 0, -1, 0, 1, 1, 0]  # h
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (w=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (w=1)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (w=2)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (w=3)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
