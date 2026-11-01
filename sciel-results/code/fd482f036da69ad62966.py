"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Rank-7 Strassen decomposition for 2x2 matrix multiplication.
    # This is the known theoretical minimum rank for 2x2 matrix multiplication.
    # The score is determined by the rank, so 7 is the optimal value.
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products defined as (u_coeffs, v_coeffs, w_coeffs)
    # Indices for U (A): 0:A00, 1:A01, 2:A10, 3:A11
    # Indices for V (B): 0:B00, 1:B01, 2:B10, 3:B11
    # Indices for W (C): 0:C00, 1:C10, 2:C01, 3:C11 (using W[k*n+i])
    
    # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
    U[0][0] += 1.0; U[3][0] += 1.0
    V[0][0] += 1.0; V[3][0] += 1.0
    W[0][0] += 1.0; W[3][0] += 1.0
    
    # M2 = (A10 + A11)B00 -> C10 - C11
    U[2][1] += 1.0; U[3][1] += 1.0
    V[0][1] += 1.0
    W[1][1] += 1.0; W[3][1] += -1.0
    
    # M3 = A00(B01 - B11) -> C01 + C11
    U[0][2] += 1.0
    V[1][2] += 1.0; V[3][2] += -1.0
    W[2][2] += 1.0; W[3][2] += 1.0
    
    # M4 = A11(B10 - B00) -> C00 + C10
    U[3][3] += 1.0
    V[2][3] += 1.0; V[0][3] += -1.0
    W[0][3] += 1.0; W[1][3] += 1.0
    
    # M5 = (A00 + A01)B11 -> -C00 + C01
    U[0][4] += 1.0; U[1][4] += 1.0
    V[3][4] += 1.0
    W[0][4] += -1.0; W[2][4] += 1.0
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2][5] += 1.0; U[0][5] += -1.0
    V[0][5] += 1.0; V[1][5] += 1.0
    W[3][5] += 1.0
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1][6] += 1.0; U[3][6] += -1.0
    V[2][6] += 1.0; V[3][6] += 1.0
    W[0][6] += 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
