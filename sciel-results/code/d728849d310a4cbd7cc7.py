"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 45 decomposition
    # Part 1: 2x1x10 (Rank 10)
    # Part 2: 2x2x10 using 5 Strassen 2x2x2 blocks (Rank 35)
    rank = 10 + 5 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A[:, 0] * B[0, :]
    # U rows: i*m + j. j=0.
    # V rows: j*p + k. j=0.
    # W rows: k*n + i.
    for k in range(p):
        c = k
        # A[0,0]
        U[0 * m + 0][c] += 1
        # A[1,0]
        U[1 * m + 0][c] += 1
        # B[0,k]
        V[0 * p + k][c] += 1
        # C[0,k]
        W[k * n + 0][c] += 1
        # C[1,k]
        W[k * n + 1][c] += 1
        
    # Part 2: A[:, 1:3] * B[1:3, :]
    # A sub-matrix:
    # A[0,1] -> A00
    # A[0,2] -> A01
    # A[1,1] -> A10
    # A[1,2] -> A11
    #
    # B sub-matrix (block b, cols 2b, 2b+1):
    # B[1, 2b]   -> B00
    # B[1, 2b+1] -> B01
    # B[2, 2b]   -> B10
    # B[2, 2b+1] -> B11
    #
    # C sub-matrix (block b, cols 2b, 2b+1):
    # C[0, 2b]   -> C00
    # C[0, 2b+1] -> C01
    # C[1, 2b]   -> C10
    # C[1, 2b+1] -> C11

    for b in range(5):
        base_col = 10 + b * 7
        for s in range(7):
            c = base_col + s
            
            if s == 0:
                # M0 = (A00 + A11)(B00 + B11)
                U[0 * m + 1][c] += 1  # A00
                U[1 * m + 2][c] += 1  # A11
                V[1 * p + 2*b][c] += 1    # B00
                V[2 * p + 2*b+1][c] += 1  # B11
                W[2*b * n + 0][c] += 1    # C00
                W[(2*b+1) * n + 1][c] += 1 # C11
                
            elif s == 1:
                # M1 = (A10 + A11)B00
                U[1 * m + 1][c] += 1  # A10
                U[1 * m + 2][c] += 1  # A11
                V[1 * p + 2*b][c] += 1    # B00
                W[2*b * n + 1][c] += 1    # C10
                W[(2*b+1) * n + 1][c] += -1 # C11
                
            elif s == 2:
                # M2 = A00(B10 - B11)
                U[0 * m + 1][c] += 1  # A00
                V[2 * p + 2*b][c] += 1    # B10
                V[2 * p + 2*b+1][c] += -1 # B11
                W[(2*b+1) * n + 0][c] += 1 # C01
                W[(2*b+1) * n + 1][c] += 1 # C11
                
            elif s == 3:
                # M3 = A11(B01 - B00)
                U[1 * m + 2][c] += 1  # A11
                V[1 * p + 2*b+1][c] += 1 # B01
                V[1 * p + 2*b][c] += -1   # B00
                W[2*b * n + 0][c] += 1    # C00
                W[2*b * n + 1][c] += 1    # C10
                
            elif s == 4:
                # M4 = (A00 + A01)B11
                U[0 * m + 1][c] += 1  # A00
                U[0 * m + 2][c] += 1  # A01
                V[2 * p + 2*b+1][c] += 1 # B11
                W[2*b * n + 0][c] += -1   # C00
                W[(2*b+1) * n + 0][c] += 1 # C01
                
            elif s == 5:
                # M5 = (A10 - A00)(B00 + B01)
                U[1 * m + 1][c] += 1  # A10
                U[0 * m + 1][c] += -1 # A00
                V[1 * p + 2*b][c] += 1    # B00
                V[1 * p + 2*b+1][c] += 1  # B01
                W[(2*b+1) * n + 1][c] += 1 # C11
                
            elif s == 6:
                # M6 = (A01 - A11)(B10 + B11)
                U[0 * m + 2][c] += 1  # A01
                U[1 * m + 2][c] += -1 # A11
                V[2 * p + 2*b][c] += 1    # B10
                V[2 * p + 2*b+1][c] += 1  # B11
                W[2*b * n + 0][c] += 1    # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
