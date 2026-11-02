"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # This is the proven optimal rank for 2x2 matmul.
    #
    # Conventions:
    #   U has shape (n*m, rank) = (4, 7), indexed U[i*m + j, c]
    #   V has shape (m*p, rank) = (4, 7), indexed V[j*p + k, c]
    #   W has shape (p*n, rank) = (4, 7), indexed W[k*n + i, c]
    #
    # The decomposition satisfies:
    #   (A*B)[i,k] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    #   where the sum over c runs over the rank-7 terms.
    #
    # Strassen's 7 products:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # Result:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    n, m, p = DIMENSIONS
    rank = 7
    
    # U[i*m + j, c] - entries from matrix A
    # Row indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    U = [
        # c:  0    1    2    3    4    5    6
        [1,   0,   1,   0,   1,   1,   1],   # A00 (i=0,j=0)
        [0,   0,   0,   0,   1,   0,  -1],   # A01 (i=0,j=1)
        [0,   1,   0,   0,   0,  -1,   0],   # A10 (i=1,j=0)
        [1,   1,   0,   1,   0,   0,   0],   # A11 (i=1,j=1)
    ]
    
    # V[j*p + k, c] - entries from matrix B
    # Row indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
    V = [
        # c:  0    1    2    3    4    5    6
        [1,   1,   0,  -1,   0,   1,   0],   # B00 (j=0,k=0)
        [0,   0,   1,   0,   0,   1,   0],   # B01 (j=0,k=1)
        [1,   0,  -1,   1,   0,   0,   1],   # B10 (j=1,k=0)
        [1,   0,   0,   0,   1,   0,   1],   # B11 (j=1,k=1)
    ]
    
    # W[k*n + i, c] - entries for output C[i,k]
    # Row indices: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    W = [
        # c:  0    1    2    3    4    5    6
        [1,   0,   0,   1,  -1,   0,   1],   # C00 (k=0,i=0)
        [0,   1,   0,   0,   0,   0,   0],   # C10 (k=0,i=1)
        [0,   0,   1,   0,   1,   0,   0],   # C01 (k=1,i=0)
        [1,  -1,   1,   0,   0,   1,   0],   # C11 (k=1,i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
