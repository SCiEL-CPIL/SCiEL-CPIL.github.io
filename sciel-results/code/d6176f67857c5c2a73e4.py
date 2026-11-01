"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21: 3 blocks of 2x2x2, each using Strassen's rank 7 algorithm
    rank = 21
    
    # U: shape (n*m, rank) = (4, 21)
    # V: shape (m*p, rank) = (12, 21)
    # W: shape (p*n, rank) = (12, 21)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block b (cols 2b, 2b+1):
    #   0:(0, 2b), 1:(0, 2b+1), 2:(1, 2b), 3:(1, 2b+1) relative to j*p+k
    #   Global V indices:
    #   v00 = 0*6 + 2b = 2b
    #   v01 = 0*6 + 2b+1 = 2b+1
    #   v10 = 1*6 + 2b = 6+2b
    #   v11 = 1*6 + 2b+1 = 6+2b+1
    # W indices for block b (cols 2b, 2b+1):
    #   W index = k*n + i = 2k + i
    #   w00 (C_0, 2b) = 2(2b) + 0 = 4b
    #   w01 (C_0, 2b+1) = 2(2b+1) + 0 = 4b+2
    #   w10 (C_1, 2b) = 2(2b) + 1 = 4b+1
    #   w11 (C_1, 2b+1) = 2(2b+1) + 1 = 4b+3
    
    for b in range(3):
        # Block start column in rank
        col_start = b * 7
        
        # V global indices for this block
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 6 + 2 * b
        v11 = 6 + 2 * b + 1
        
        # W global indices for this block
        w00 = 4 * b
        w01 = 4 * b + 2
        w10 = 4 * b + 1
        w11 = 4 * b + 3
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col_start] = 1
        U[3][col_start] = 1
        V[v00][col_start] = 1
        V[v11][col_start] = 1
        W[w00][col_start] = 1
        W[w10][col_start] = -1
        W[w01][col_start] = -1
        W[w11][col_start] = 1
        
        # M2 = (A10 + A11) * B00
        U[2][col_start + 1] = 1
        U[3][col_start + 1] = 1
        V[v00][col_start + 1] = 1
        W[w10][col_start + 1] = 1
        W[w11][col_start + 1] = -1
        
        # M3 = A00 * (B01 - B11)
        U[0][col_start + 2] = 1
        V[v01][col_start + 2] = 1
        V[v11][col_start + 2] = -1
        W[w01][col_start + 2] = 1
        W[w11][col_start + 2] = 1
        
        # M4 = A11 * (B10 - B00)
        U[3][col_start + 3] = 1
        V[v10][col_start + 3] = 1
        V[v00][col_start + 3] = -1
        W[w00][col_start + 3] = 1
        
        # M5 = (A00 + A01) * B11
        U[0][col_start + 4] = 1
        U[1][col_start + 4] = 1
        V[v11][col_start + 4] = 1
        W[w00][col_start + 4] = -1
        W[w01][col_start + 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col_start + 5] = 1
        U[0][col_start + 5] = -1
        V[v00][col_start + 5] = 1
        V[v01][col_start + 5] = 1
        W[w11][col_start + 5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col_start + 6] = 1
        U[3][col_start + 6] = -1
        V[v10][col_start + 6] = 1
        V[v11][col_start + 6] = 1
        W[w00][col_start + 6] = -1
        W[w10][col_start + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
