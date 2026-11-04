import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen Basis for 2x2x2
    # u, v, w are length 4 vectors (2x2 matrices flattened)
    strassen_basis = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    
    # Generate 49 terms for 4x4x4 part
    # Outer level (2x2 blocks) x Inner level (2x2 elements)
    for r_outer in range(7):
        u_outer, v_outer, w_outer = strassen_basis[r_outer]
        for r_inner in range(7):
            u_inner, v_inner, w_inner = strassen_basis[r_inner]
            
            # Fill U (4x4)
            for I in range(2):
                for J in range(2):
                    val_o = u_outer[I * 2 + J]
                    if val_o != 0:
                        for ip in range(2):
                            for jp in range(2):
                                val_i = u_inner[ip * 2 + jp]
                                if val_i != 0:
                                    i = 2 * I + ip
                                    j = 2 * J + jp
                                    U[i * m + j][col] = val_o * val_i
                                    
            # Fill V (4x4, cols 0-3)
            for J in range(2):
                for K in range(2):
                    val_o = v_outer[J * 2 + K]
                    if val_o != 0:
                        for jp in range(2):
                            for kp in range(2):
                                val_i = v_inner[jp * 2 + kp]
                                if val_i != 0:
                                    j = 2 * J + jp
                                    k = 2 * K + kp
                                    V[j * p + k][col] = val_o * val_i
                                    
            # Fill W (4x4, cols 0-3)
            for K in range(2):
                for I in range(2):
                    val_o = w_outer[K * 2 + I]
                    if val_o != 0:
                        for kp in range(2):
                            for ip in range(2):
                                val_i = w_inner[kp * 2 + ip]
                                if val_i != 0:
                                    k = 2 * K + kp
                                    i = 2 * I + ip
                                    W[k * n + i][col] = val_o * val_i
                                    
            col += 1
            
    # Add 16 schoolbook terms for 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
