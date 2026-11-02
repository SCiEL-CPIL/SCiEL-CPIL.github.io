"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 10 decomposition
    rank = 10
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen rank 7 for i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Mapping to U, V, W columns 0..6
    
    # Col 0: M1
    U[0*3+0][0] = 1 # A00
    U[1*3+1][0] = 1 # A11
    V[0*4+0][0] = 1 # B00
    V[1*4+1][0] = 1 # B11
    W[0*2+0][0] = 1 # C00
    W[1*2+0][0] = 1 # C10
    W[0*2+1][0] = 1 # C01
    W[1*2+1][0] = 1 # C11
    
    # Col 1: M2
    U[1*3+0][1] = 1 # A10
    U[1*3+1][1] = 1 # A11
    V[0*4+0][1] = 1 # B00
    W[0*2+1][1] = 1 # C10
    
    # Col 2: M3
    U[0*3+0][2] = 1 # A00
    V[0*4+1][2] = 1 # B01
    V[1*4+1][2] = -1 # B11
    W[1*2+0][2] = 1 # C01
    
    # Col 3: M4
    U[1*3+1][3] = 1 # A11
    V[1*4+0][3] = 1 # B10
    V[0*4+0][3] = -1 # B00
    W[0*2+1][3] = 1 # C10
    
    # Col 4: M5
    U[0*3+0][4] = 1 # A00
    U[0*3+1][4] = 1 # A01
    V[1*4+1][4] = 1 # B11
    W[1*2+1][4] = 1 # C11
    
    # Col 5: M6
    U[1*3+0][5] = 1 # A10
    U[0*3+0][5] = -1 # A00
    V[0*4+0][5] = 1 # B00
    V[0*4+1][5] = 1 # B01
    W[1*2+1][5] = 1 # C11
    
    # Col 6: M7
    U[0*3+1][6] = 1 # A01
    U[1*3+1][6] = -1 # A11
    V[1*4+0][6] = 1 # B10
    V[1*4+1][6] = 1 # B11
    W[0*2+0][6] = 1 # C00
    
    # 2. Rank 1 for j in {0,1}, k=2
    # T[i, j, 2] = 1
    # U[i*3+j, 7] = 1
    # V[j*4+2, 7] = 1
    # W[2*2+i, 7] = 1
    for i in range(2):
        for j in range(2):
            U[i*3+j][7] = 1
            V[j*4+2][7] = 1
            W[2*2+i][7] = 1
            
    # 3. Rank 1 for j in {0,1}, k=3
    # T[i, j, 3] = 1
    # U[i*3+j, 8] = 1
    # V[j*4+3, 8] = 1
    # W[3*2+i, 8] = 1
    for i in range(2):
        for j in range(2):
            U[i*3+j][8] = 1
            V[j*4+3][8] = 1
            W[3*2+i][8] = 1
            
    # 4. Rank 1 for j=2, k in {0,1,2,3}
    # T[i, 2, k] = 1
    # U[i*3+2, 9] = 1
    # V[2*4+k, 9] = 1
    # W[k*2+i, 9] = 1
    for i in range(2):
        for k in range(4):
            U[i*3+2][9] = 1
            V[2*4+k][9] = 1
            W[k*2+i][9] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
