"""
Rank 33 decomposition for 2x3 by 3x6 matrix multiplication.
Uses Strassen's algorithm for 2x2x2 blocks and outer product for the remaining part.
"""

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Outer product for a_{i2} * b_{2k} (Rank 12)
    # r = 0..11
    for i in range(2):
        for k in range(6):
            r = i * 6 + k
            U[i * 3 + 2][r] = 1
            V[2 * 6 + k][r] = 1
            W[k * 2 + i][r] = 1
            
    # Part 2: Strassen for B block cols 0,1 (Rank 7)
    # r = 12..18
    # A = [[a00, a01], [a10, a11]] -> U indices 0,1,3,4
    # B = [[b00, b01], [b10, b11]] -> V indices 0,1,6,7
    # C outputs: C00 (W0), C01 (W2), C10 (W1), C11 (W3)
    
    # M1 = (a00+a11)(b00+b11)
    r = 12
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[7][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # M2 = (a10+a11)b00
    r = 13
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    # M3 = a00(b01-b11)
    r = 14
    U[0][r] = 1
    V[1][r] = 1; V[7][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    # M4 = a11(b10-b00)
    r = 15
    U[4][r] = 1
    V[6][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # M5 = (a00+a01)b11
    r = 16
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # M6 = (a10-a00)(b00+b01)
    r = 17
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # M7 = (a01-a11)(b10+b11)
    r = 18
    U[1][r] = 1; U[4][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[0][r] = 1
    
    # Part 3: Strassen for B block cols 2,3 (Rank 7)
    # r = 19..25
    # B = [[b02, b03], [b12, b13]] -> V indices 2,3,8,9
    # C outputs: C02 (W4), C03 (W6), C12 (W5), C13 (W7)
    
    # M1
    r = 19
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[9][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    # M2
    r = 20
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    # M3
    r = 21
    U[0][r] = 1
    V[3][r] = 1; V[9][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    # M4
    r = 22
    U[4][r] = 1
    V[8][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    # M5
    r = 23
    U[0][r] = 1; U[1][r] = 1
    V[9][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    # M6
    r = 24
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    # M7
    r = 25
    U[1][r] = 1; U[4][r] = -1
    V[8][r] = 1; V[9][r] = 1
    W[4][r] = 1
    
    # Part 4: Strassen for B block cols 4,5 (Rank 7)
    # r = 26..32
    # B = [[b04, b05], [b14, b15]] -> V indices 4,5,10,11
    # C outputs: C04 (W8), C05 (W10), C14 (W9), C15 (W11)
    
    # M1
    r = 26
    U[0][r] = 1; U[4][r] = 1
    V[4][r] = 1; V[11][r] = 1
    W[8][r] = 1; W[11][r] = 1
    
    # M2
    r = 27
    U[3][r] = 1; U[4][r] = 1
    V[4][r] = 1
    W[9][r] = 1; W[11][r] = -1
    
    # M3
    r = 28
    U[0][r] = 1
    V[5][r] = 1; V[11][r] = -1
    W[10][r] = 1; W[11][r] = 1
    
    # M4
    r = 29
    U[4][r] = 1
    V[10][r] = 1; V[4][r] = -1
    W[8][r] = 1; W[9][r] = 1
    
    # M5
    r = 30
    U[0][r] = 1; U[1][r] = 1
    V[11][r] = 1
    W[8][r] = -1; W[10][r] = 1
    
    # M6
    r = 31
    U[3][r] = 1; U[0][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[11][r] = 1
    
    # M7
    r = 32
    U[1][r] = 1; U[4][r] = -1
    V[10][r] = 1; V[11][r] = 1
    W[8][r] = 1

    return {"U": U, "V": V, "W": W}
