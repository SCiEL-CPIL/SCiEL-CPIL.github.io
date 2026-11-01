import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # s=0..6
    # U_coeffs[s] is tuple of 4 values for a00, a01, a10, a11
    # V_coeffs[s] is tuple of 4 values for b00, b01, b10, b11
    # W_coeffs[s] is tuple of 4 values for c00, c01, c10, c11
    
    U_coeffs = [
        (1, 0, 0, 1),   # M1: a00 + a11
        (0, 0, 1, 1),   # M2: a10 + a11
        (1, 0, 0, 0),   # M3: a00
        (0, 0, 0, 1),   # M4: a11
        (1, 1, 0, 0),   # M5: a00 + a01
        (-1, 0, 1, 0),  # M6: a10 - a00
        (0, 1, 0, -1)   # M7: a01 - a11
    ]
    
    V_coeffs = [
        (1, 0, 0, 1),   # M1: b00 + b11
        (1, 0, 0, 0),   # M2: b00
        (0, 1, 0, -1),  # M3: b01 - b11
        (-1, 0, 1, 0),  # M4: b10 - b00
        (0, 0, 0, 1),   # M5: b11
        (1, 1, 0, 0),   # M6: b00 + b01
        (0, 0, 1, 1)    # M7: b10 + b11
    ]
    
    W_coeffs = [
        (1, 0, 0, 1),   # M1: c00 + c11
        (0, 0, 1, -1),  # M2: c10 - c11
        (0, 1, 0, 1),   # M3: c01 + c11
        (1, 0, 1, 0),   # M4: c00 + c10
        (-1, 1, 0, 0),  # M5: -c00 + c01
        (0, 0, 0, 1),   # M6: c11
        (1, 0, 0, 0)    # M7: c00
    ]
    
    # Mapping local indices to global indices
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    for s in range(7):
        for b in range(7):
            col = s * 7 + b
            
            # Fill U
            # U rows: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            U[0][col] = U_coeffs[s][0]
            U[1][col] = U_coeffs[s][1]
            U[2][col] = U_coeffs[s][2]
            U[3][col] = U_coeffs[s][3]
            
            # Fill V
            # V rows for block b:
            # b00 -> 0*p + 2b = 2b
            # b01 -> 0*p + 2b+1 = 2b+1
            # b10 -> 1*p + 2b = 14+2b
            # b11 -> 1*p + 2b+1 = 14+2b+1
            V[2*b][col] = V_coeffs[s][0]
            V[2*b+1][col] = V_coeffs[s][1]
            V[14+2*b][col] = V_coeffs[s][2]
            V[14+2*b+1][col] = V_coeffs[s][3]
            
            # Fill W
            # W rows for block b:
            # c00 -> (2b)*2 + 0 = 4b
            # c01 -> (2b+1)*2 + 0 = 4b+2
            # c10 -> (2b)*2 + 1 = 4b+1
            # c11 -> (2b+1)*2 + 1 = 4b+3
            W[4*b][col] = W_coeffs[s][0]
            W[4*b+2][col] = W_coeffs[s][1]
            W[4*b+1][col] = W_coeffs[s][2]
            W[4*b+3][col] = W_coeffs[s][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
