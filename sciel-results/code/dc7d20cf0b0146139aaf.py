"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=3
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (6, rank)
    # W: (p*n, rank) = (6, rank)
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # Strassen decomposition for B[:, 0:2]
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1  # B00 (0*3+0), B11 (1*3+1)
    W[0][0] = 1; W[3][0] = 1  # C00 (0*2+0), C11 (1*2+1)
    
    # Col 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1 # C10 (0*2+1), C11 (1*2+1)
    
    # Col 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1 # B01 (0*3+1), B11 (1*3+1)
    W[2][2] = 1; W[3][2] = -1 # C01 (1*2+0), C11 (1*2+1)
    
    # Col 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1 # B10 (1*3+0), B00 (0*3+0)
    W[0][3] = -1; W[1][3] = 1 # C00 (0*2+0), C10 (0*2+1)
    
    # Col 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[2][4] = 1; W[3][4] = 1 # C01, C11
    
    # Col 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = 1; W[2][5] = -1 # C00, C10, -C01
    
    # Col 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[1][6] = -1; W[2][6] = 1 # -C00, -C10, C01
    
    # Standard decomposition for B[:, 2]
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Col 7: A00 * B02 -> C02
    U[0][7] = 1
    V[2][7] = 1 # B02 (0*3+2)
    W[4][7] = 1 # C02 (2*2+0)
    
    # Col 8: A01 * B12 -> C02
    U[1][8] = 1
    V[5][8] = 1 # B12 (1*3+2)
    W[4][8] = 1 # C02
    
    # Col 9: A10 * B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1 # C12 (2*2+1)
    
    # Col 10: A11 * B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
