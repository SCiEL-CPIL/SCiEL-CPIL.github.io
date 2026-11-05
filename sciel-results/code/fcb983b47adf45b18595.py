import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 89: 7 for Strassen (2x2x2 sub-block) + 82 for remaining schoolbook terms
    rank = 89
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for the 2x2x2 sub-block
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    # U indices (i*m+j):
    # A00: 0*5+0 = 0
    # A01: 0*5+1 = 1
    # A10: 1*5+0 = 5
    # A11: 1*5+1 = 6
    # V indices (j*p+k):
    # B00: 0*6+0 = 0
    # B01: 0*6+1 = 1
    # B10: 1*6+0 = 6
    # B11: 1*6+1 = 7
    # W indices (k*n+i):
    # C00: 0*3+0 = 0
    # C01: 0*3+1 = 1
    # C10: 1*3+0 = 3
    # C11: 1*3+1 = 4
    
    # Strassen Coefficients
    # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
    U[0][0] = 1; U[6][0] = 1
    V[0][0] = 1; V[7][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # M2 = (A10 + A11)B00 -> C10 - C11
    U[5][1] = 1; U[6][1] = 1
    V[0][1] = 1
    W[3][1] = 1; W[4][1] = -1
    
    # M3 = A00(B10 - B11) -> C01 + C11
    U[0][2] = 1
    V[6][2] = 1; V[7][2] = -1
    W[1][2] = 1; W[4][2] = 1
    
    # M4 = A11(B01 - B00) -> C00 + C10
    U[6][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[3][3] = 1
    
    # M5 = (A00 + A01)B11 -> -C00 + C01
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[5][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1][6] = 1; U[6][6] = -1
    V[6][6] = 1; V[7][6] = 1
    W[0][6] = 1
    
    # Remaining schoolbook terms
    # Tuples (i, j, k) NOT in {0,1}x{0,1}x{0,1}
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this tuple is in the Strassen sub-block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify col == 89
    # 90 total - 8 skipped = 82. 7 + 82 = 89.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
