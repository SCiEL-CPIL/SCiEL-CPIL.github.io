"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 (cols 0-6)
    # Indices:
    # U: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # V: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1) 4:(0,2) 5:(1,2)
    # W: 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1) 4:(0,2) 5:(1,2)
    
    # Col 0: M1
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2
    U[0][1] = 1; U[2][1] = 1
    V[3][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Col 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Col 3: M4
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # Col 4: M5
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Col 5: M6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Col 6: M7
    U[1][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    # Schoolbook for 3rd column (cols 7-10)
    # Col 7: A00 B02 -> C02
    U[0][7] = 1
    V[4][7] = 1
    W[4][7] = 1
    
    # Col 8: A01 B12 -> C02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Col 9: A10 B02 -> C12
    U[2][9] = 1
    V[4][9] = 1
    W[5][9] = 1
    
    # Col 10: A11 B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
