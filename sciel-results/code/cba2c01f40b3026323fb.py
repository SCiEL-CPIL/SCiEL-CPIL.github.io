import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :] using Strassen blocks
    for b in range(5):
        k0 = 2 * b
        
        # M1 = (A00 + A11)(B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[1][col] += 1
        V[10 + k0 + 1][col] += 1
        V[20 + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[5][col] += 1
        V[20 + k0][col] += 1
        V[10 + k0][col] -= 1
        W[2 * k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[20 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
