import time

def construct():
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (cols 0-6) for k in {0, 1}
    # U rows: i*2+j
    # V rows: j*3+k
    # W rows: k*2+i
    
    # Col 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Col 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Col 3: M4
    U[3][3] = 1
    V[3][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Col 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Col 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Col 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook part (cols 7-10) for k = 2
    # Col 7: A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Col 8: A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Col 9: A10 * B02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Col 10: A11 * B12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
