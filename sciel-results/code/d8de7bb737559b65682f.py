import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    
    # Output Z:
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We define the index mappings for each of the 4 blocks
    # Each block provides a mapping:
    # x_idx: {11: u_idx, 12: u_idx, 21: u_idx, 22: u_idx}
    # y_idx: {11: v_idx, 12: v_idx, 21: v_idx, 22: v_idx}
    # z_idx: {11: w_idx, 12: w_idx, 21: w_idx, 22: w_idx}
    
    blocks = [
        # Block 1: A1 B11 -> C[:, 0:2]
        {
            'x': {11: 0, 12: 1, 21: 4, 22: 5},
            'y': {11: 0, 12: 1, 21: 4, 22: 5},
            'z': {11: 0, 12: 2, 21: 1, 22: 3}
        },
        # Block 2: A2 B21 -> C[:, 0:2]
        {
            'x': {11: 2, 12: 3, 21: 6, 22: 7},
            'y': {11: 8, 12: 9, 21: 12, 22: 13},
            'z': {11: 0, 12: 2, 21: 1, 22: 3}
        },
        # Block 3: A1 B12 -> C[:, 2:4]
        {
            'x': {11: 0, 12: 1, 21: 4, 22: 5},
            'y': {11: 2, 12: 3, 21: 6, 22: 7},
            'z': {11: 4, 12: 6, 21: 5, 22: 7}
        },
        # Block 4: A2 B22 -> C[:, 2:4]
        {
            'x': {11: 2, 12: 3, 21: 6, 22: 7},
            'y': {11: 10, 12: 11, 21: 14, 22: 15},
            'z': {11: 4, 12: 6, 21: 5, 22: 7}
        }
    ]
    
    col = 0
    for b in blocks:
        x = b['x']
        y = b['y']
        z = b['z']
        
        # M1
        U[x[11]][col] += 1
        U[x[22]][col] += 1
        V[y[11]][col] += 1
        V[y[22]][col] += 1
        W[z[11]][col] += 1
        W[z[22]][col] += 1
        col += 1
        
        # M2
        U[x[21]][col] += 1
        U[x[22]][col] += 1
        V[y[11]][col] += 1
        W[z[21]][col] += 1
        W[z[22]][col] -= 1
        col += 1
        
        # M3
        U[x[11]][col] += 1
        V[y[12]][col] += 1
        V[y[22]][col] -= 1
        W[z[12]][col] += 1
        W[z[22]][col] += 1
        col += 1
        
        # M4
        U[x[22]][col] += 1
        V[y[21]][col] += 1
        V[y[11]][col] -= 1
        W[z[11]][col] += 1
        W[z[21]][col] += 1
        col += 1
        
        # M5
        U[x[11]][col] += 1
        U[x[12]][col] += 1
        V[y[22]][col] += 1
        W[z[11]][col] -= 1
        W[z[12]][col] += 1
        col += 1
        
        # M6
        U[x[21]][col] += 1
        U[x[11]][col] -= 1
        V[y[11]][col] += 1
        V[y[12]][col] += 1
        W[z[22]][col] += 1
        col += 1
        
        # M7
        U[x[12]][col] += 1
        U[x[22]][col] -= 1
        V[y[21]][col] += 1
        V[y[22]][col] += 1
        W[z[11]][col] += 1
        W[z[21]][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
