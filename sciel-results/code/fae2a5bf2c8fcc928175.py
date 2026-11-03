# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's algorithm for 3x3x3
    # ...
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
