import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u, v, w are lists of length 4 (flattened 2x2)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for t in range(7):
        u, v, w = strassen_terms[t]
        for i in range(2):
            for j in range(2):
                U[i * 4 + j][col] += u[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[j * 4 + k][col] += v[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][col] += w[k * 2 + i]
        col += 1
        
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for t in range(7):
        u, v, w = strassen_terms[t]
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2)][col] += u[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * 4 + k][col] += v[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][col] += w[k * 2 + i]
        col += 1
        
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for t in range(7):
        u, v, w = strassen_terms[t]
        for i in range(2):
            for j in range(2):
                U[i * 4 + j][col] += u[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[j * 4 + (k + 2)][col] += v[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i][col] += w[k * 2 + i]
        col += 1
        
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    for t in range(7):
        u, v, w = strassen_terms[t]
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2)][col] += u[i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * 4 + (k + 2)][col] += v[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i][col] += w[k * 2 + i]
        col += 1
        
    return {"U": U, "V": V, "W": W}

