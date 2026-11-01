"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    # U, V, W are lists of lists (matrices) with shapes (4, 7), (4, 7), (4, 7).
    
    # U rows correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows correspond to C[0,0], C[1,0], C[0,1], C[1,1] (column-major flattening)
    
    U = [
        [0, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [1, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]
    
    V = [
        [1, 0, 1, -1, 0, 1, 0],  # B[0,0]
        [1, 0, 0, 1, 0, 0, 0],   # B[0,1]
        [0, 1, -1, 0, 0, 1, 1],  # B[1,0]
        [0, 0, 0, 0, 1, 0, 1]    # B[1,1]
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
