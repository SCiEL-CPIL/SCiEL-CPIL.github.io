"""Strassen rank-7 tensor decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: 4 x 7 (n*m = 4 rows, rank=7 cols)
    # U row index: i*m + j
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # i=0,j=0
        [0,  0,  0,  0,  1,  0,  1],  # i=0,j=1
        [0,  1,  0,  0,  0,  1,  0],  # i=1,j=0
        [1,  1,  0,  1,  0,  0, -1],  # i=1,j=1
    ]
    
    # V: 4 x 7 (m*p = 4 rows, rank=7 cols)
    # V row index: j*p + k
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # j=0,k=0
        [0,  0,  1,  0,  0,  1,  0],  # j=0,k=1
        [0,  0,  0,  1,  0,  0,  1],  # j=1,k=0
        [1, -1, -1,  0,  1,  0,  1],  # j=1,k=1
    ]
    
    # W: 4 x 7 (p*n = 4 rows, rank=7 cols)
    # W row index: k*n + i
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # k=0,i=0
        [0,  0,  1,  0,  1,  0,  0],  # k=0,i=1
        [0,  1,  0,  1,  0,  0,  0],  # k=1,i=0
        [1, -1,  1,  0,  0,  1,  0],  # k=1,i=1
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
