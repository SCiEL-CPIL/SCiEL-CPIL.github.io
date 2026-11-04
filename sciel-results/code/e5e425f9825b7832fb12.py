import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 blocks
    # A rows 0,1 cols 0,1
    # B rows 0,1 cols 4j, 4j+1
    # C rows 0,1 cols 4j, 4j+1
    
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    for j in range(4):
        b00 = 0 + 4 * j
        b01 = 1 + 4 * j
        b10 = 8 + 4 * j
        b11 = 9 + 4 * j
        
        c00 = 3 * (4 * j) + 0
        c01 = 3 * (4 * j + 1) + 0
        c10 = 3 * (4 * j) + 1
        c11 = 3 * (4 * j + 1) + 1
        
        col = j * 7
        
        # M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        
        # M2
        U[a10][col+1] += 1
        U[a11][col+1] += 1
        V[b00][col+1] += 1
        W[c10][col+1] += 1
        
        # M3
        U[a00][col+2] += 1
        V[b01][col+2] += 1
        V[b11][col+2] += -1
        W[c01][col+2] += 1
        
        # M4
        U[a11][col+3] += 1
        V[b10][col+3] += 1
        V[b00][col+3] += -1
        W[c00][col+3] += 1
        W[c10][col+3] += 1
        
        # M5
        U[a00][col+4] += 1
        U[a01][col+4] += 1
        V[b11][col+4] += 1
        W[c00][col+4] += -1
        W[c01][col+4] += 1
        
        # M6
        U[a10][col+5] += 1
        U[a00][col+5] += -1
        V[b00][col+5] += 1
        V[b01][col+5] += 1
        W[c11][col+5] += 1
        
        # M7
        U[a01][col+6] += 1
        U[a11][col+6] += -1
        V[b10][col+6] += 1
        V[b11][col+6] += 1
        W[c00][col+6] += 1
        
    # Term 2: A[0,2] B[2,k] -> C[0,k]
    col = 28
    U[2][col] = 1
    for k in range(8):
        V[16 + k][col] = 1
        W[3 * k][col] = 1
        
    # Term 2: A[1,2] B[2,k] -> C[1,k]
    col = 29
    U[5][col] = 1
    for k in range(8):
        V[16 + k][col] = 1
        W[3 * k + 1][col] = 1
        
    # Term 3: A[2,0] B[0,k] + A[2,1] B[1,k] -> C[2,k]
    col = 30
    U[6][col] = 0.5
    U[7][col] = 0.5
    for k in range(8):
        V[k][col] = 0.5
        V[8 + k][col] = 0.5
        W[3 * k + 2][col] = 2
        
    # Term 4: A[2,2] B[2,k] -> C[2,k]
    col = 31
    U[8][col] = 1
    for k in range(8):
        V[16 + k][col] = 1
        W[3 * k + 2][col] = 1
        
    return {"U": U, "V": V, "W": W}
