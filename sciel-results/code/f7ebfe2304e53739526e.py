import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 3, 3, 8
    rank = 47

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1a: Strassen for i in {0,1}, j in {0,1}, k in {0..7}
    # 7 terms
    # Strassen formulas for C_2x2 = A_2x2 * B_2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Coefficients for C
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6

    # We define U, V, W for each M_r
    # U is 3x3, V is 3x8, W is 8x3
    
    # M1
    U[0*3+0][col] = 1
    U[1*3+1][col] = 1
    for k in range(p):
        V[0*8+k][col] = 1
        V[1*8+k][col] = 1
        W[k*3+0][col] = 1
        W[k*3+1][col] = 1
    col += 1

    # M2
    U[1*3+0][col] = 1
    U[1*3+1][col] = 1
    for k in range(p):
        V[0*8+k][col] = 1
        W[k*3+1][col] = 1
    col += 1

    # M3
    U[0*3+0][col] = 1
    for k in range(p):
        V[0*8+k][col] = 1
        V[1*8+k][col] = -1
        W[k*3+0][col] = 1
    col += 1

    # M4
    U[1*3+1][col] = 1
    for k in range(p):
        V[1*8+k][col] = 1
        V[0*8+k][col] = -1
        W[k*3+1][col] = 1
    col += 1

    # M5
    U[0*3+0][col] = 1
    U[0*3+1][col] = 1
    for k in range(p):
        V[1*8+k][col] = 1
        W[k*3+0][col] = 1
    col += 1

    # M6
    U[1*3+0][col] = 1
    U[0*3+0][col] = -1
    for k in range(p):
        V[0*8+k][col] = 1
        W[k*3+1][col] = 1
    col += 1

    # M7
    U[0*3+1][col] = 1
    U[1*3+1][col] = -1
    for k in range(p):
        V[1*8+k][col] = 1
        W[k*3+0][col] = 1
    col += 1

    # Part 1b: i in {0,1}, j=2, k in {0..7}
    # 16 terms
    for i in [0, 1]:
        for k in range(p):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+i][col] = 1
            col += 1

    # Part 2a: i=2, j in {0,1}, k in {0..7}
    # 16 terms
    for j in [0, 1]:
        for k in range(p):
            U[2*3+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1

    # Part 2b: i=2, j=2, k in {0..7}
    # 8 terms
    for k in range(p):
        U[2*3+2][col] = 1
        V[2*8+k][col] = 1
        W[k*3+2][col] = 1
        col += 1

    if col != rank:
        print(f"Error: col {col} != rank {rank}")
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
