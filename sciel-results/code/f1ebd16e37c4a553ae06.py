import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for the 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # Rank of block = 7
    # Rank of rest = (3*4*5) - (2*2*2) = 60 - 8 = 52
    # Total Rank = 59
    
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Block
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1: a00 * b00 -> c00 += M1, c01 += 0, c10 += 0, c11 += 0 ? 
    # Strassen Output:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M2 + M5
    # c10 = M3 + M6
    # c11 = M1 - M2 + M4 + M6
    
    # Term 1: a00 * b00
    # U: a00 (i=0, j=0) -> idx 0*4+0 = 0. Val 1
    # V: b00 (j=0, k=0) -> idx 0*5+0 = 0. Val 1
    # W: c00 (k=0, i=0) -> idx 0*3+0 = 0. Val 1
    # W: c01 (k=1, i=0) -> idx 1*3+0 = 3. Val 0
    # W: c10 (k=0, i=1) -> idx 0*3+1 = 1. Val 0
    # W: c11 (k=1, i=1) -> idx 1*3+1 = 4. Val 0
    # Wait, Strassen formula:
    # c00 uses M1, M4, M5, M7
    # c01 uses M2, M5
    # c10 uses M3, M6
    # c11 uses M1, M2, M4, M6
    
    # M1 = a00 * b00
    U[0][col] = 1
    V[0][col] = 1
    W[0][col] = 1   # c00
    W[4][col] = 1   # c11 (Wait, c11 uses M1? Yes. c11 = M1 - M2 + M4 + M6)
    col += 1
    
    # M2 = a00 * (b10 + b11)
    U[0][col] = 1
    V[5][col] = 1   # b10 (j=1, k=0) -> 1*5+0=5
    V[6][col] = 1   # b11 (j=1, k=1) -> 1*5+1=6
    W[3][col] = 1   # c01 (k=1, i=0) -> 1*3+0=3
    W[4][col] = -1  # c11 (k=1, i=1) -> 1*3+1=4
    col += 1
    
    # M3 = (a10 + a11) * b00
    U[4][col] = 1   # a10 (i=1, j=0) -> 1*4+0=4
    U[5][col] = 1   # a11 (i=1, j=1) -> 1*4+1=5
    V[0][col] = 1   # b00
    W[1][col] = 1   # c10 (k=0, i=1) -> 0*3+1=1
    col += 1
    
    # M4 = a11 * b11
    U[5][col] = 1
    V[6][col] = 1
    W[0][col] = 1   # c00
    W[4][col] = 1   # c11
    col += 1
    
    # M5 = (a00 + a11) * (b00 + b11)
    U[0][col] = 1
    U[5][col] = 1
    V[0][col] = 1
    V[6][col] = 1
    W[0][col] = -1  # c00
    W[3][col] = 1   # c01
    col += 1
    
    # M6 = (a10 - a00) * (b00 + b01)
    U[4][col] = 1
    U[0][col] = -1
    V[0][col] = 1
    V[1][col] = 1   # b01 (j=0, k=1) -> 0*5+1=1
    W[1][col] = 1   # c10
    W[4][col] = 1   # c11
    col += 1
    
    # M7 = (a01 + a11) * (b10 - b00)
    U[1][col] = 1   # a01 (i=0, j=1) -> 0*4+1=1
    U[5][col] = 1
    V[5][col] = 1   # b10
    V[0][col] = -1  # b00
    W[0][col] = 1   # c00
    col += 1
    
    # Schoolbook Rest
    # All (i,j,k) NOT in i{0,1}, j{0,1}, k{0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
