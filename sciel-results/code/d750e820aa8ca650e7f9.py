import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # For p=2, V[j*2+k], W[k*2+i]
    
    # Strassen coefficients
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b00-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U (A factors)
    # a00, a01, a10, a11
    # M1: a00=1, a11=1
    # M2: a10=1, a11=1
    # M3: a00=1
    # M4: a11=1
    # M5: a00=1, a01=1
    # M6: a10=1, a00=-1
    # M7: a01=1, a11=-1
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # V (B factors) for p=2
    # b00, b01, b10, b11
    # M1: b00=1, b11=1
    # M2: b00=1
    # M3: b10=1, b11=-1
    # M4: b00=1, b01=-1
    # M5: b11=1
    # M6: b00=1, b01=1
    # M7: b10=1, b11=1
    
    V_str = [
        [1, 1, 0, 1, 0, 1, 0], # b00
        [0, 0, 0, -1, 0, 1, 0], # b01
        [0, 0, 1, 0, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    # W (Output factors) for p=2, n=2
    # c00, c01, c10, c11
    # c00 = M1 - M4 - M6 + M7
    # c01 = M3 + M5 + M7
    # c10 = M2 + M4 + M6
    # c11 = M1 - M3 + M5
    
    W_str = [
        [1, 0, 0, -1, 0, -1, 1], # c00
        [0, 0, 1, 0, 1, 0, 1],   # c01
        [0, 1, 0, 1, 0, 1, 0],   # c10
        [1, 0, -1, 0, 1, 0, 0]   # c11
    ]
    
    # Schoolbook 2x2x1 factors
    # a00, a01, a10, a11
    # b00, b10
    # c00 = a00 b00
    # c01 = a01 b00? No, c01 is row 0 col 1?
    # For 2x2x1, B is 2x1.
    # C is 2x1.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    
    U_sb = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    
    V_sb = [
        [1, 0, 0, 0], # b0
        [0, 1, 0, 0]  # b1
    ]
    
    W_sb = [
        [1, 0, 0, 0], # c0
        [0, 0, 1, 0]  # c1
    ]
    # Wait, W_sb indices: k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # For 2x2x1, p=1.
    # W is 2x1 matrix? No, W is vector of length p*n = 2.
    # W[k*n+i].
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # So W_sb[0] is c0, W_sb[1] is c1.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # So W_sb should be:
    # W_sb[0] (c0) gets 1 from a00 b0 and a01 b1.
    # W_sb[1] (c1) gets 1 from a10 b0 and a11 b1.
    
    W_sb = [
        [1, 1, 0, 0], # c0
        [0, 0, 1, 1]  # c1
    ]
    
    # Construct U, V, W
    # Block 1: cols 0,1. Rank 7.
    # U: cols 0-6
    # V: rows 0,1,7,8 (j*7+k)
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    # W: rows 0,1,2,3 (k*2+i)
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    for c in range(7):
        for i in range(4):
            U[i][c] = U_str[i][c]
        V[0][c] = V_str[0][c] # b00
        V[1][c] = V_str[1][c] # b01
        V[7][c] = V_str[2][c] # b10
        V[8][c] = V_str[3][c] # b11
        W[0][c] = W_str[0][c] # c00
        W[1][c] = W_str[1][c] # c01
        W[2][c] = W_str[2][c] # c10
        W[3][c] = W_str[3][c] # c11
        
    # Block 2: cols 2,3. Rank 7.
    # U: cols 7-13
    # V: rows 2,3,9,10
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 9
    # j=1, k=3 -> 10
    # W: rows 4,5,6,7
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    
    for c in range(7):
        for i in range(4):
            U[i][7+c] = U_str[i][c]
        V[2][7+c] = V_str[0][c]
        V[3][7+c] = V_str[1][c]
        V[9][7+c] = V_str[2][c]
        V[10][7+c] = V_str[3][c]
        W[4][7+c] = W_str[0][c]
        W[5][7+c] = W_str[1][c]
        W[6][7+c] = W_str[2][c]
        W[7][7+c] = W_str[3][c]
        
    # Block 3: cols 4,5. Rank 7.
    # U: cols 14-20
    # V: rows 4,5,11,12
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=1, k=4 -> 11
    # j=1, k=5 -> 12
    # W: rows 8,9,10,11
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    # k=5, i=0 -> 10
    # k=5, i=1 -> 11
    
    for c in range(7):
        for i in range(4):
            U[i][14+c] = U_str[i][c]
        V[4][14+c] = V_str[0][c]
        V[5][14+c] = V_str[1][c]
        V[11][14+c] = V_str[2][c]
        V[12][14+c] = V_str[3][c]
        W[8][14+c] = W_str[0][c]
        W[9][14+c] = W_str[1][c]
        W[10][14+c] = W_str[2][c]
        W[11][14+c] = W_str[3][c]
        
    # Block 4: col 6. Rank 4.
    # U: cols 21-24
    # V: rows 6, 13
    # j=0, k=6 -> 6
    # j=1, k=6 -> 13
    # W: rows 12, 13
    # k=6, i=0 -> 12
    # k=6, i=1 -> 13
    
    for c in range(4):
        for i in range(4):
            U[i][21+c] = U_sb[i][c]
        V[6][21+c] = V_sb[0][c]
        V[13][21+c] = V_sb[1][c]
        W[12][21+c] = W_sb[0][c]
        W[13][21+c] = W_sb[1][c]
        
    return {"U": U, "V": V, "W": W}
