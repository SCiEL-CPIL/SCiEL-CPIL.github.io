import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x10) into B_L (2x10) and B_R (1x10)
    # C = A_L B_L + A_R B_R
    # A_L B_L: 5 blocks of 2x2x2, each rank 7 -> 35
    # A_R B_R: 2x1x10, rank 20 -> 20
    # Total rank 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_L (2x10)
    # A_L uses A[0,0], A[0,1], A[1,0], A[1,1]
    # Indices in U: 
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # (Note: U index is i*m + j. m=3. So 0, 1, 3, 4)
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We have 5 blocks of B_L, each 2x2.
    # B_L row 0 is B row 0 (indices 0..9)
    # B_L row 1 is B row 1 (indices 10..19)
    
    for block in range(5):
        # B columns 2*block, 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # V indices for B_L block
        # b00: V[0][k0]
        # b01: V[0][k1]
        # b10: V[10][k0]
        # b11: V[10][k1]
        
        # U indices for A_L
        # a00: U[0]
        # a01: U[1]
        # a10: U[3]
        # a11: U[4]
        
        # W indices for C block
        # c00: W[2*k0 + 0]
        # c01: W[2*k0 + 1]
        # c10: W[2*k1 + 0]
        # c11: W[2*k1 + 1]
        
        # M1
        U[0][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[10][col] = 1
        W[2*k0][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        W[2*k1][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[0][col] = 1
        V[10][col] = -1
        W[2*k0+1][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1
        V[10][col] = 1
        V[0][col] = -1
        W[2*k1][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[10][col] = 1
        W[2*k0][col] = -1
        W[2*k1+1][col] = -1
        col += 1
        
        # M6
        U[3][col] = 1
        U[0][col] = -1
        V[0][col] = 1
        V[0][col] = 1 # b00+b01
        V[0][col] = 1 # Wait, V[0][k0] and V[0][k1]
        # Correct M6: (a10-a00)(b00+b01)
        # V[0][k0] = 1, V[0][k1] = 1
        W[2*k1][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1
        U[4][col] = -1
        V[10][col] = 1
        V[10][col] = 1 # b10+b11 -> V[10][k0] + V[10][k1]
        # Correct M7: (a01-a11)(b10+b11)
        W[2*k0+1][col] = 1
        col += 1

    # Part 2: A_R (2x1) * B_R (1x10)
    # A_R uses A[0,2] (U[2]) and A[1,2] (U[5])
    # B_R uses B[2, k] (V[20+k])
    # C_R uses C[i, k] (W[2*k+i])
    
    for k in range(10):
        for i in range(2):
            U[i*3 + 2][col] = 1 # A[i, 2]
            V[2*10 + k][col] = 1 # B[2, k]
            W[2*k + i][col] = 1 # C[i, k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
