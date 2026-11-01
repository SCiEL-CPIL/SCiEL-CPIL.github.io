"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-29 decomposition.
    # Split m=3 into m1=2, m2=1.
    # A is 2x3, split into A1 (2x2) and A2 (2x1).
    # B is 3x5, split into B1 (2x5) and B2 (1x5).
    # C = A1*B1 + A2*B2.
    
    # A2*B2 is 2x1 * 1x5. Rank 10.
    # A1*B1 is 2x2 * 2x5.
    # Split B1 into B11 (2x2) and B12 (2x3).
    # A1*B11 is 2x2 * 2x2. Rank 7 (Strassen).
    # A1*B12 is 2x2 * 2x3. Rank 12 (Schoolbook).
    # Total Rank = 10 + 7 + 12 = 29.
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A2 (2x1) * B2 (1x5)
    # A2 indices: i in 0..1, j in 2 (since m1=2, j2=2)
    # B2 indices: j in 2, k in 0..4
    # U[i*m + 2, c] = A_i2
    # V[2*p + k, c] = B_2k
    # W[k*n + i, c] = C_ik
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 10.
    
    # Part 2: A1 (2x2) * B11 (2x2) using Strassen
    # A1 indices: i in 0..1, j in 0..1
    # B11 indices: j in 0..1, k in 0..1
    # Strassen 7 products.
    # M1 = A11*(B11+B22) -> (0,0) * ((0,0)+(1,1))
    # M2 = (A21+A22)*B11 -> (1,0)+(1,1) * (0,0)
    # M3 = A11*(B12-B22) -> (0,0) * ((0,1)-(1,1))
    # M4 = A22*(B21-B11) -> (1,1) * ((1,0)-(0,0))
    # M5 = (A11+A22)*(B11+B22) -> ((0,0)+(1,1)) * ((0,0)+(1,1))
    # M6 = (A12-A22)*(B21+B22) -> ((0,1)-(1,1)) * ((1,0)+(1,1))
    # M7 = (A11-A12)*(B11+B12) -> ((0,0)-(0,1)) * ((0,0)+(0,1))
    
    # Note: B12 in Strassen context is B_11 matrix's (0,1) block, which is scalar B_01 in 2x2.
    # Here B11 is 2x2.
    # Let's define the 7 terms explicitly.
    
    # Term 1: A_00 * (B_00 + B_11)
    U[0 * m + 0][col] = 1
    V[0 * p + 0][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # Term 2: (A_10 + A_11) * B_00
    U[1 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    
    # Term 3: A_00 * (B_01 - B_11)
    U[0 * m + 0][col] = 1
    V[0 * p + 1][col] = 1
    V[1 * p + 1][col] = -1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # Term 4: A_11 * (B_10 - B_00)
    U[1 * m + 1][col] = 1
    V[1 * p + 0][col] = 1
    V[0 * p + 0][col] = -1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    
    # Term 5: (A_00 + A_11) * (B_00 + B_11)
    U[0 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # Term 6: (A_01 - A_11) * (B_10 + B_11)
    U[0 * m + 1][col] = 1
    U[1 * m + 1][col] = -1
    V[1 * p + 0][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # Term 7: (A_00 - A_01) * (B_00 + B_01)
    U[0 * m + 0][col] = 1
    U[0 * m + 1][col] = -1
    V[0 * p + 0][col] = 1
    V[0 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    
    # col is now 17.
    
    # Part 3: A1 (2x2) * B12 (2x3) Schoolbook
    # A1 indices: i in 0..1, j in 0..1
    # B12 indices: j in 0..1, k in 2..4 (since B11 was 0..1)
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 17 + 12 = 29.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
