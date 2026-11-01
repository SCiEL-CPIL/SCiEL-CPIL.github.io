"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: (n*m, rank) -> (4, 7)
    # Rows correspond to a00, a01, a10, a11
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # a00
        [0,  0,  0,  0,  1,  0,  1],  # a01
        [0,  1,  0,  0,  0,  1,  0],  # a10
        [1,  1,  0,  1,  0,  0, -1]   # a11
    ]
    
    # V: (m*p, rank) -> (4, 7)
    # Rows correspond to b00, b01, b10, b11
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # b00
        [0,  0,  1,  0,  0,  1,  0],  # b01
        [0,  0,  0,  1,  0,  0,  1],  # b10
        [1,  0, -1,  0,  1,  0,  1]   # b11
    ]
    
    # W: (p*n, rank) -> (4, 7)
    # Rows correspond to C00, C01, C10, C11 (index k*n+i)
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00 (k=0, i=0)
        [0,  0,  1,  0,  1,  0,  0],  # C01 (k=1, i=0)
        [0,  1,  0,  1,  0,  0,  0],  # C10 (k=0, i=1)
        [1, -1,  1,  0,  0,  1,  0]   # C11 (k=1, i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
