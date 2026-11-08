import time
import sys

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # A is n x m (4x4)
    # B is m x p (4x8)
    # C is n x p (4x8)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # U coefficients for X (2x2)
    # Indices in X: 00, 01, 10, 11
    U_coeffs = [
        [1, 0, 0, 1],   # M1 x
        [0, 0, 1, 1],   # M2 x
        [1, 0, 0, 0],   # M3 x
        [0, 0, 0, 1],   # M4 x
        [1, 1, 0, 0],   # M5 x
        [-1, 0, 1, 0],  # M6 x
        [0, 1, 0, -1]   # M7 x
    ]
    
    # V coefficients for Y (2x2)
    # Indices in Y: 00, 01, 10, 11
    V_coeffs = [
        [1, 0, 0, 1],   # M1 y
        [1, 0, 0, 0],   # M2 y
        [0, 1, 0, -1],  # M3 y
        [0, 0, 1, -1],  # M4 y
        [0, 0, 0, 1],   # M5 y
        [1, 1, 0, 0],   # M6 y
        [0, 0, 1, 1]    # M7 y
    ]
    
    # W coefficients for Z (2x2)
    # Indices in Z: 00, 01, 10, 11
    W_coeffs = [
        [1, 0, 0, 1],   # M1 -> Z00, Z11
        [0, 0, 1, -1],  # M2 -> Z10, -Z11
        [0, 1, 0, 1],   # M3 -> Z01, Z11
        [1, 0, 1, 0],   # M4 -> Z00, Z10
        [-1, 1, 0, 0],  # M5 -> -Z00, Z01
        [0, 0, 0, 1],   # M6 -> Z11
        [1, 0, 0, 0]    # M7 -> Z00
    ]
    
    col = 0
    for I in range(2):
        for J in range(4):
            # Block A_I0: rows 2I, 2I+1; cols 0, 1
            # Block A_I1: rows 2I, 2I+1; cols 2, 3
            # Block B_0J: rows 0, 1; cols 2J, 2J+1
            # Block B_1J: rows 2, 3; cols 2J, 2J+1
            # Block C_IJ: rows 2I, 2I+1; cols 2J, 2J+1
            
            # Term 1: A_I0 * B_0J
            for k in range(7):
                # U part
                for r in range(2):
                    for c in range(2):
                        u_idx = (2*I + r) * m + c
                        U[u_idx][col] += U_coeffs[k][r*2+c]
                        
                # V part
                for r in range(2):
                    for c in range(2):
                        v_idx = r * p + (2*J + c)
                        V[v_idx][col] += V_coeffs[k][r*2+c]
                        
                # W part
                for r in range(2):
                    for c in range(2):
                        # C index is (2I+r, 2J+c)
                        # W index is (2J+c)*n + (2I+r)
                        w_idx = (2*J + c) * n + (2*I + r)
                        W[w_idx][col] += W_coeffs[k][r*2+c]
                        
                col += 1
                
            # Term 2: A_I1 * B_1J
            for k in range(7):
                # U part
                for r in range(2):
                    for c in range(2):
                        u_idx = (2*I + r) * m + (2 + c)
                        U[u_idx][col] += U_coeffs[k][r*2+c]
                        
                # V part
                for r in range(2):
                    for c in range(2):
                        v_idx = (2 + r) * p + (2*J + c)
                        V[v_idx][col] += V_coeffs[k][r*2+c]
                        
                # W part
                for r in range(2):
                    for c in range(2):
                        # C index is (2I+r, 2J+c)
                        # W index is (2J+c)*n + (2I+r)
                        w_idx = (2*J + c) * n + (2*I + r)
                        W[w_idx][col] += W_coeffs[k][r*2+c]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
