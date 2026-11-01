"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    # Column 0: M1 = (A00 + A11)(B00 + B11)
    # U indices: A00 (0*2+0=0), A11 (1*2+1=3)
    # V indices: B00 (0*2+0=0), B11 (1*2+1=3)
    # W contributions to C00 (0*2+0=0), C11 (1*2+1=3)
    U[0][0] = 1.0
    U[3][0] = 1.0
    V[0][0] = 1.0
    V[3][0] = 1.0
    W[0][0] = 1.0   # C00
    W[3][0] = 1.0   # C11
    
    # Column 1: M2 = (A10 + A11)B00
    # U indices: A10 (1*2+0=2), A11 (1*2+1=3)
    # V indices: B00 (0*2+0=0)
    # W contributions to C10 (1*2+0=2), C11 (1*2+1=3)
    U[2][1] = 1.0
    U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0   # C10
    W[3][1] = -1.0  # C11
    
    # Column 2: M3 = A00(B01 - B11)
    # U indices: A00 (0*2+0=0)
    # V indices: B01 (0*2+1=1), B11 (1*2+1=3)
    # W contributions to C01 (0*2+1=1), C11 (1*2+1=3)
    U[0][2] = 1.0
    V[1][2] = 1.0
    V[3][2] = -1.0
    W[1][2] = 1.0   # C01
    W[3][2] = 1.0   # C11
    
    # Column 3: M4 = A11(B10 - B00)
    # U indices: A11 (1*2+1=3)
    # V indices: B10 (1*2+0=2), B00 (0*2+0=0)
    # W contributions to C00 (0*2+0=0), C10 (1*2+0=2)
    U[3][3] = 1.0
    V[2][3] = 1.0
    V[0][3] = -1.0
    W[0][3] = 1.0   # C00
    W[2][3] = 1.0   # C10
    
    # Column 4: M5 = (A00 + A01)B11
    # U indices: A00 (0*2+0=0), A01 (0*2+1=1)
    # V indices: B11 (1*2+1=3)
    # W contributions to C00 (0*2+0=0), C01 (0*2+1=1)
    U[0][4] = 1.0
    U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0  # C00
    W[1][4] = 1.0   # C01
    
    # Column 5: M6 = (A10 - A00)(B00 + B01)
    # U indices: A10 (1*2+0=2), A00 (0*2+0=0)
    # V indices: B00 (0*2+0=0), B01 (0*2+1=1)
    # W contributions to C11 (1*2+1=3)
    U[2][5] = 1.0
    U[0][5] = -1.0
    V[0][5] = 1.0
    V[1][5] = 1.0
    W[3][5] = 1.0   # C11
    
    # Column 6: M7 = (A01 - A11)(B10 + B11)
    # U indices: A01 (0*2+1=1), A11 (1*2+1=3)
    # V indices: B10 (1*2+0=2), B11 (1*2+1=3)
    # W contributions to C00 (0*2+0=0)
    U[1][6] = 1.0
    U[3][6] = -1.0
    V[2][6] = 1.0
    V[3][6] = 1.0
    W[0][6] = 1.0   # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
