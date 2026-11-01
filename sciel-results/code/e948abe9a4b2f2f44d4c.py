import numpy as np
import time

def construct():
    # Theoretical minimum rank for M_{2,2,8} is 4 * Rank(M_{2,2,2}) = 4 * 7 = 28.
    # We construct the decomposition using Strassen's algorithm for each of the 4 
    # 2x2 blocks of the B matrix.
    
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U corresponds to A coefficients (A00, A01, A10, A11)
    # V corresponds to B coefficients (B00, B01, B10, B11)
    # W corresponds to C coefficients (C00, C01, C10, C11)
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    num_blocks = p // 2 # 4 blocks
    
    for t in range(num_blocks):
        for r in range(7):
            # Fill U
            U[0, col] = U_s[r][0]
            U[1, col] = U_s[r][1]
            U[2, col] = U_s[r][2]
            U[3, col] = U_s[r][3]
            
            # Fill V
            # B indices: B[row, col] -> row * p + col
            # Block t covers columns 2t and 2t+1
            v00 = 0 * p + 2 * t
            v01 = 0 * p + (2 * t + 1)
            v10 = 1 * p + 2 * t
            v11 = 1 * p + (2 * t + 1)
            
            V[v00, col] = V_s[r][0]
            V[v01, col] = V_s[r][1]
            V[v10, col] = V_s[r][2]
            V[v11, col] = V_s[r][3]
            
            # Fill W
            # C indices: W[k*n+i, r] where k is col, i is row
            # C indices: C[i, k] -> k * n + i
            w00 = (2 * t) * n + 0
            w01 = (2 * t + 1) * n + 0
            w10 = (2 * t) * n + 1
            w11 = (2 * t + 1) * n + 1
            
            W[w00, col] = W_s[r][0]
            W[w01, col] = W_s[r][1]
            W[w10, col] = W_s[r][2]
            W[w11, col] = W_s[r][3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
