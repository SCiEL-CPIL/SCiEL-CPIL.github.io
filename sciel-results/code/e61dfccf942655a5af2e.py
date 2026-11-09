import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    # Part 1: 4x4 * 4x4 Strassen, rank 49
    for s_out in range(7):
        for s_in in range(7):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # U vector: 16 elements, index = i*4 + j
            for i in range(4):
                for j in range(4):
                    i_block = i // 2
                    j_block = j // 2
                    i_local = i % 2
                    j_local = j % 2
                    u_val = u_in[s_out][i_block*2+j_block] * u_in[s_in][i_local*2+j_local]
                    U[i*4+j][col] = u_val
            
            # V vector: 28 elements, index = j*7 + k (but only k=0..3 for this part)
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    k_block = k // 2
                    j_local = j % 2
                    k_local = k % 2
                    v_val = v_in[s_out][j_block*2+k_block] * v_in[s_in][j_local*2+k_local]
                    V[j*7+k][col] = v_val
            
            # W vector: 28 elements, index = k*4 + i
            for i in range(4):
                for k in range(4):
                    i_block = i // 2
                    k_block = k // 2
                    i_local = i % 2
                    k_local = k % 2
                    w_val = w_out[k_block*2+i_block][s_out] * w_out[k_local*2+i_local][s_in]
                    W[k*4+i][col] = w_val
            
            col += 1
    
    # Part 2: 4x4 * 4x3 schoolbook, rank 48
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                kg = 4 + kp
                U[i * 4 + j][col] = 1
                V[j * 7 + kg][col] = 1
                W[kg * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
