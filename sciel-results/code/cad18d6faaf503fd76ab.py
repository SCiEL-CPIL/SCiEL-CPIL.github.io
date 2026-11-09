import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(u_offs, v_offs, w_offs, start_col):
        nonlocal col
        # u_offs: [u00, u01, u10, u11]
        # v_offs: [v00, v01, v10, v11]
        # w_offs: [w00, w01, w10, w11]
        
        # M1
        # X: 1, 0, 0, 1
        # Y: 1, 0, 0, 1
        # Z: 1 (to 00), 0 (to 01), 0 (to 10), 1 (to 11)
        U[u_offs[0]][start_col] += 1
        U[u_offs[3]][start_col] += 1
        V[v_offs[0]][start_col] += 1
        V[v_offs[3]][start_col] += 1
        W[w_offs[0]][start_col] += 1
        W[w_offs[3]][start_col] += 1
        col += 1
        
        # M2
        # X: 0, 0, 1, 1
        # Y: 1, 0, 0, 0
        # Z: 0 (to 00), 0 (to 01), 1 (to 10), -1 (to 11)
        U[u_offs[2]][start_col] += 1
        U[u_offs[3]][start_col] += 1
        V[v_offs[0]][start_col] += 1
        W[w_offs[2]][start_col] += 1
        W[w_offs[3]][start_col] -= 1
        col += 1
        
        # M3
        # X: 1, 0, 0, 0
        # Y: 0, 1, -1, 0
        # Z: 0 (to 00), 1 (to 01), 0 (to 10), 1 (to 11)
        U[u_offs[0]][start_col] += 1
        V[v_offs[1]][start_col] += 1
        V[v_offs[2]][start_col] -= 1
        W[w_offs[1]][start_col] += 1
        W[w_offs[3]][start_col] += 1
        col += 1
        
        # M4
        # X: 0, 0, 0, 1
        # Y: 0, -1, 1, 0
        # Z: 1 (to 00), 0 (to 01), 1 (to 10), 0 (to 11)
        U[u_offs[3]][start_col] += 1
        V[v_offs[1]][start_col] -= 1
        V[v_offs[2]][start_col] += 1
        W[w_offs[0]][start_col] += 1
        W[w_offs[2]][start_col] += 1
        col += 1
        
        # M5
        # X: 1, 1, 0, 0
        # Y: 0, 0, 0, 1
        # Z: -1 (to 00), 1 (to 01), 0 (to 10), 0 (to 11)
        U[u_offs[0]][start_col] += 1
        U[u_offs[1]][start_col] += 1
        V[v_offs[3]][start_col] += 1
        W[w_offs[0]][start_col] -= 1
        W[w_offs[1]][start_col] += 1
        col += 1
        
        # M6
        # X: -1, 0, 1, 0
        # Y: 1, 1, 0, 0
        # Z: 0 (to 00), 0 (to 01), 0 (to 10), 1 (to 11)
        U[u_offs[0]][start_col] -= 1
        U[u_offs[2]][start_col] += 1
        V[v_offs[0]][start_col] += 1
        V[v_offs[1]][start_col] += 1
        W[w_offs[3]][start_col] += 1
        col += 1
        
        # M7
        # X: 0, -1, 0, 1
        # Y: 0, 0, 1, 1
        # Z: 1 (to 00), 0 (to 01), 0 (to 10), 0 (to 11)
        U[u_offs[1]][start_col] -= 1
        U[u_offs[3]][start_col] += 1
        V[v_offs[2]][start_col] += 1
        V[v_offs[3]][start_col] += 1
        W[w_offs[0]][start_col] += 1
        col += 1

    # Part 1: 2x4x4 (Rank 28)
    
    # Block 1: A_L B_T1
    # A_L: U[0], U[1], U[4], U[5]
    # B_T1: V[0], V[1], V[4], V[5]
    # C: W[0], W[1], W[4], W[5]
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], col)
    
    # Block 2: A_L B_T2
    # A_L: U[0], U[1], U[4], U[5]
    # B_T2: V[2], V[3], V[6], V[7]
    # C: W[2], W[3], W[6], W[7]
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7], col)
    
    # Block 3: A_R B_B1
    # A_R: U[2], U[3], U[6], U[7]
    # B_B1: V[8], V[9], V[12], V[13]
    # C: W[0], W[1], W[4], W[5]
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5], col)
    
    # Block 4: A_R B_B2
    # A_R: U[2], U[3], U[6], U[7]
    # B_B2: V[10], V[11], V[14], V[15]
    # C: W[2], W[3], W[6], W[7]
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7], col)
    
    # Part 2: 1x4x4 (Rank 16)
    for k in range(4):
        for j in range(4):
            U[8 + k][col] = 1
            V[k * 4 + j][col] = 1
            W[j * 3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
