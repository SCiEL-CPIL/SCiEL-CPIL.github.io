import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We are constructing a rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2 multiplication
    # Each step is (U_coeffs, V_coeffs, W_coeffs)
    # Local indices: 0:00, 1:01, 2:10, 3:11
    # U_coeffs: [a00, a01, a10, a11]
    # V_coeffs: [b00, b01, b10, b11]
    # W_coeffs: [c00, c01, c10, c11]
    
    steps = [
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for u_l, v_l, w_l in steps:
                        # Map local A indices to global
                        # Local (r, c) in block (a, b)
                        # Global row = 2a + r, Global col = 2b + c
                        # Local index 0 -> (0,0) -> (2a, 2b)
                        # Local index 1 -> (0,1) -> (2a, 2b+1)
                        # Local index 2 -> (1,0) -> (2a+1, 2b)
                        # Local index 3 -> (1,1) -> (2a+1, 2b+1)
                        
                        for li in range(4):
                            r = li // 2
                            c_loc = li % 2
                            gi = 2 * a + r
                            gj = 2 * b + c_loc
                            U[gi * 4 + gj][col] += u_l[li]
                            
                            r = li // 2
                            c_loc = li % 2
                            gj2 = 2 * c + r
                            gk = 2 * d + c_loc
                            V[gj2 * 4 + gk][col] += v_l[li]
                            
                            r = li // 2
                            c_loc = li % 2
                            gi2 = 2 * a + r
                            gk2 = 2 * d + c_loc
                            W[gk2 * 4 + gi2][col] += w_l[li]
                        
                        col += 1
                        
    return {"U": U, "V": V, "W": W}
