import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 35 decomposition using Strassen's algorithm on 5 blocks of 2x2 matrices.
    R = 35
    U = [[0] * R for _ in range(n * m)]  # 4 x 35
    V = [[0] * R for _ in range(m * p)]  # 20 x 35
    W = [[0] * R for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U indices: i*m + j. For 2x2, indices are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k. For a block q (columns 2q, 2q+1):
    #   j=0, k=0 -> 0*10 + 2q = 2q
    #   j=0, k=1 -> 0*10 + 2q+1 = 2q+1
    #   j=1, k=0 -> 1*10 + 2q = 10+2q
    #   j=1, k=1 -> 1*10 + 2q+1 = 10+2q+1
    # W indices: k*n + i. For a block q:
    #   k=2q, i=0 -> 4q
    #   k=2q, i=1 -> 4q+1
    #   k=2q+1, i=0 -> 4q+2
    #   k=2q+1, i=1 -> 4q+3
    
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    col = 0
    for q in range(5):
        for u_v, v_v, w_v in strassen_terms:
            # Fill U
            for i in range(4):
                U[i][col] = u_v[i]
            
            # Fill V
            V[2*q][col] = v_v[0]
            V[2*q+1][col] = v_v[1]
            V[10+2*q][col] = v_v[2]
            V[10+2*q+1][col] = v_v[3]
            
            # Fill W
            W[4*q][col] = w_v[0]
            W[4*q+1][col] = w_v[1]
            W[4*q+2][col] = w_v[2]
            W[4*q+3][col] = w_v[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
