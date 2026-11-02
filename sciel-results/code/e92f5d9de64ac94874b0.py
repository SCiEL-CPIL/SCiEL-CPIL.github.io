import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 pairs
    for k in range(0, 14, 2):
        # M1
        U[0][col] += 1
        U[3][col] += 1
        V[k][col] += 1
        V[15+k+1][col] += 1
        W[2*k][col] += 1
        W[2*k+3][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1
        U[3][col] += 1
        V[k][col] += 1
        W[2*k+1][col] += 1
        W[2*k+3][col] += -1
        col += 1
        
        # M3
        U[0][col] += 1
        V[k+1][col] += 1
        V[15+k+1][col] += -1
        W[2*k+2][col] += 1
        W[2*k+3][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[15+k][col] += 1
        V[k][col] += -1
        W[2*k][col] += 1
        W[2*k+1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1
        U[1][col] += 1
        V[15+k+1][col] += 1
        W[2*k][col] += -1
        W[2*k+2][col] += 1
        col += 1
        
        # M6
        U[0][col] += -1
        U[2][col] += 1
        V[k][col] += 1
        V[k+1][col] += 1
        W[2*k+3][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1
        U[3][col] += -1
        V[15+k][col] += 1
        V[15+k+1][col] += 1
        W[2*k][col] += 1
        col += 1

    # Single column k=14
    # A00 B0,14
    U[0][col] += 1
    V[14][col] += 1
    W[28][col] += 1
    col += 1
    
    # A01 B1,14
    U[1][col] += 1
    V[29][col] += 1
    W[28][col] += 1
    col += 1
    
    # A10 B0,14
    U[2][col] += 1
    V[14][col] += 1
    W[29][col] += 1
    col += 1
    
    # A11 B1,14
    U[3][col] += 1
    V[29][col] += 1
    W[29][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
