"""Optimized program for 2x2x12 matrix multiplication using Strassen blocks."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],   # S0: a00 + a11
        [0, 0, 1, 1],   # S1: a10 + a11
        [1, 0, 0, 0],   # S2: a00
        [0, 0, 0, 1],   # S3: a11
        [1, 1, 0, 0],   # S4: a00 + a01
        [-1, 0, 1, 0],  # S5: a10 - a00
        [0, 1, 0, -1]   # S6: a01 - a11
    ]
    
    # For each of 6 column-pairs u and each Strassen form t
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V: B is 2x12, indexed by row (0,1) and col (0..11)
            # B index: row * p + col, so row0: 0..11, row1: 12..23
            # For column pair u, columns are 2u and 2u+1
            b00 = 0 * p + 2*u       # B[0][2u]
            b01 = 0 * p + 2*u + 1   # B[0][2u+1]
            b10 = 1 * p + 2*u       # B[1][2u]
            b11 = 1 * p + 2*u + 1   # B[1][2u+1]
            
            if t == 0:   # b00 + b11
                V[b00][c] += 1
                V[b11][c] += 1
            elif t == 1: # b00
                V[b00][c] += 1
            elif t == 2: # b01 - b11
                V[b01][c] += 1
                V[b11][c] -= 1
            elif t == 3: # b10 - b00
                V[b10][c] += 1
                V[b00][c] -= 1
            elif t == 4: # b11
                V[b11][c] += 1
            elif t == 5: # b00 + b01
                V[b00][c] += 1
                V[b01][c] += 1
            elif t == 6: # b10 + b11
                V[b10][c] += 1
                V[b11][c] += 1
                
            # Fill W: C is 2x12, indexed W[k*n + i] where k is col index, i is row index
            # C row 0, col 2u:   M0 + M3 - M4 + M6
            # C row 0, col 2u+1: M2 + M4
            # C row 1, col 2u:   M1 + M3
            # C row 1, col 2u+1: M0 - M1 + M2 + M5
            
            # W index: col * n + row = (2u or 2u+1) * 2 + (0 or 1)
            w00 = (2*u) * n + 0       # C[0][2u]
            w01 = (2*u + 1) * n + 0   # C[0][2u+1]
            w10 = (2*u) * n + 1       # C[1][2u]
            w11 = (2*u + 1) * n + 1   # C[1][2u+1]
            
            if t == 0:   # M0
                W[w00][c] += 1
                W[w11][c] += 1
            elif t == 1: # M1
                W[w01][c] += 1
                W[w11][c] -= 1
            elif t == 2: # M2
                W[w01][c] += 1
                W[w11][c] += 1
            elif t == 3: # M3
                W[w00][c] += 1
                W[w10][c] += 1
            elif t == 4: # M4
                W[w00][c] -= 1
                W[w01][c] += 1
            elif t == 5: # M5
                W[w11][c] += 1
            elif t == 6: # M6
                W[w00][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
