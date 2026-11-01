"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen for 2x2 part + Schoolbook for 2x1 part
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U[i*m+j]
    # V[j*p+k]
    # W[k*n+i]
    
    # Strassen for k=0,1 (first two columns of B)
    # M1: (a11+a22)(b11+b22) -> c11, c22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)b11 -> c21, -c22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a11(b12-b22) -> c12, c22
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a22(b21-b11) -> c11, c21
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a11+a12)b22 -> -c11, c12
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a21-a11)(b11+b12) -> c22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a12-a22)(b21+b22) -> c11
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook for k=2 (third column of B)
    # M8: a11*b13 -> c13
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # M9: a12*b23 -> c13
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # M10: a21*b13 -> c23
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # M11: a22*b23 -> c23
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
