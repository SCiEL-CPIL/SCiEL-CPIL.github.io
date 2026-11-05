import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
        (2, 0, 0),
        (2, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
    ]
    
    for i_off, j_off, k_off in blocks:
        # M1
        U[i_off*m + 0][col] += 1
        U[(i_off+1)*m + 1][col] += 1
        V[j_off*p + 0][col] += 1
        V[(j_off+1)*p + 1][col] += 1
        W[k_off*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2
        U[i_off*m + 1][col] += 1
        U[(i_off+1)*m + 1][col] += 1
        V[j_off*p + 0][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        col += 1
        
        # M3
        U[i_off*m + 0][col] += 1
        V[j_off*p + 1][col] += 1
        V[(j_off+1)*p + 1][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4
        U[(i_off+1)*m + 1][col] += 1
        V[(j_off+1)*p + 0][col] += 1
        V[j_off*p + 0][col] += -1
        W[k_off*n + i_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        col += 1
        
        # M5
        U[i_off*m + 0][col] += 1
        U[i_off*m + 1][col] += 1
        V[(j_off+1)*p + 1][col] += 1
        W[k_off*n + i_off][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        col += 1
        
        # M6
        U[i_off*m + 1][col] += 1
        U[i_off*m + 0][col] += -1
        V[j_off*p + 0][col] += 1
        V[j_off*p + 1][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7
        U[i_off*m + 1][col] += 1
        U[(i_off+1)*m + 1][col] += -1
        V[(j_off+1)*p + 0][col] += 1
        V[(j_off+1)*p + 1][col] += 1
        W[k_off*n + i_off][col] += 1
        col += 1
        
    # Schoolbook for remaining terms
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 2 and j < 2 and k < 2: continue
                if i < 2 and j >= 2 and j < 4 and k < 2: continue
                if i < 2 and j < 2 and k >= 2 and k < 4: continue
                if i < 2 and j >= 2 and j < 4 and k >= 2 and k < 4: continue
                if i >= 2 and i < 4 and j < 2 and k < 2: continue
                if i >= 2 and i < 4 and j < 2 and k >= 2 and k < 4: continue
                if i >= 2 and i < 4 and j >= 2 and j < 4 and k < 2: continue
                if i >= 2 and i < 4 and j >= 2 and j < 4 and k >= 2 and k < 4: continue
                
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
