"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank: 14 (Strassen for 2x2 x 2x4) + 8 (Outer product 2x1 x 1x4) = 22
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2x2 block
    def add_strassen(i0, k0, j0):
        nonlocal col
        # i0: start row of A (0 or 1? No, 0 or 2? A is 2x3. i is 0..1)
        # k0: start col of A (0 or 2)
        # j0: start col of B (0 or 2)
        
        # Local indices for A: (0,0), (0,1), (1,0), (1,1) -> Global (i0+r, k0+c)
        # Local indices for B: (0,0), (0,1), (1,0), (1,1) -> Global (k0+r, j0+c)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # We need to add 7 columns.
        
        # M1
        U[(i0+0)*m + (k0+0)][col] += 1
        U[(i0+1)*m + (k0+1)][col] += 1
        V[(k0+0)*p + (j0+0)][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[(j0+0)*n + (i0+0)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[(i0+1)*m + (k0+0)][col] += 1
        U[(i0+1)*m + (k0+1)][col] += 1
        V[(k0+0)*p + (j0+0)][col] += 1
        W[(j0+0)*n + (i0+1)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M3
        U[(i0+0)*m + (k0+0)][col] += 1
        V[(k0+1)*p + (j0+0)][col] += 1
        V[(k0+1)*p + (j0+1)][col] += -1
        W[(j0+1)*n + (i0+0)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4
        U[(i0+1)*m + (k0+1)][col] += 1
        V[(k0+0)*p + (j0+1)][col] += 1
        V[(k0+0)*p + (j0+0)][col] += -1
        W[(j0+0)*n + (i0+0)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[(i0+0)*m + (k0+0)][col] += 1
        U[(i0+0)*m + (k0+1)][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[(j0+0)*n + (i0+0)][col] += -1
        W[(j0+1)*n + (i0+0)][col] += 1
        col += 1
        
        # M6
        U[(i0+1)*m + (k0+0)][col] += 1
        U[(i0+0)*m + (k0+0)][col] += -1
        V[(k0+0)*p + (j0+0)][col] += 1
        V[(k0+0)*p + (j0+1)][col] += 1
        W[(j0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[(i0+0)*m + (k0+1)][col] += 1
        U[(i0+1)*m + (k0+1)][col] += -1
        V[(k0+1)*p + (j0+0)][col] += 1
        V[(k0+1)*p + (j0+1)][col] += 1
        W[(j0+0)*n + (i0+0)][col] += 1
        W[(j0+1)*n + (i0+0)][col] += -1
        col += 1

    # Part 1: A_L (rows 0-1, cols 0-1) * B_T (rows 0-1, cols 0-3)
    # Split B_T into j=0-1 and j=2-3
    add_strassen(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
    
    # Part 2: A_R (rows 0-1, col 2) * B_B (row 2, cols 0-3)
    # This is 2x1 * 1x4.
    # We do 8 scalar multiplications: A[i,2] * B[2,j] for i in 0,1 and j in 0..3
    for i in range(2):
        for j in range(4):
            # U[i*m + 2] = 1
            # V[2*p + j] = 1
            # W[j*n + i] = 1
            U[i * m + 2][col] = 1
            V[2 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
