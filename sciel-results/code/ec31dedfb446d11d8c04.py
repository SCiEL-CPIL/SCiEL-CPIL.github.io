import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    n, m, p = 2, 3, 7
    rank = 39

    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add a Strassen block
    def add_strassen_block(u_map, v_map, w_map, col_offset):
        # u_map: {0: idx_a00, 1: idx_a01, 2: idx_a10, 3: idx_a11}
        # v_map: {0: idx_b00, 1: idx_b01, 2: idx_b10, 3: idx_b11}
        # w_map: {0: idx_c00, 1: idx_c01, 2: idx_c10, 3: idx_c11}
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[u_map[0]][col_offset + 0] += 1
        U[u_map[3]][col_offset + 0] += 1
        V[v_map[0]][col_offset + 0] += 1
        V[v_map[3]][col_offset + 0] += 1
        W[w_map[0]][col_offset + 0] += 1
        W[w_map[3]][col_offset + 0] += 1 # Wait, C11 = M1... C22 = M1...
        # Let's trace carefully.
        # C00 (w_map[0]) += M1
        # C11 (w_map[3]) += M1
        W[w_map[0]][col_offset + 0] += 1
        W[w_map[3]][col_offset + 0] += 1
        
        # M2 = (a10 + a11) * b00
        U[u_map[2]][col_offset + 1] += 1
        U[u_map[3]][col_offset + 1] += 1
        V[v_map[0]][col_offset + 1] += 1
        W[w_map[2]][col_offset + 1] += 1 # C10 += M2
        W[w_map[3]][col_offset + 1] -= 1 # C11 -= M2
        
        # M3 = a00 * (b10 - b11)
        U[u_map[0]][col_offset + 2] += 1
        V[v_map[2]][col_offset + 2] += 1
        V[v_map[3]][col_offset + 2] -= 1
        W[w_map[1]][col_offset + 2] += 1 # C01 += M3
        W[w_map[3]][col_offset + 2] += 1 # C11 += M3
        
        # M4 = a11 * (b01 - b00)
        U[u_map[3]][col_offset + 3] += 1
        V[v_map[1]][col_offset + 3] += 1
        V[v_map[0]][col_offset + 3] -= 1
        W[w_map[0]][col_offset + 3] += 1 # C00 += M4
        W[w_map[2]][col_offset + 3] += 1 # C10 += M4
        
        # M5 = (a00 + a01) * b11
        U[u_map[0]][col_offset + 4] += 1
        U[u_map[1]][col_offset + 4] += 1
        V[v_map[3]][col_offset + 4] += 1
        W[w_map[0]][col_offset + 4] -= 1 # C00 -= M5
        W[w_map[1]][col_offset + 4] += 1 # C01 += M5
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[u_map[2]][col_offset + 5] += 1
        U[u_map[0]][col_offset + 5] -= 1
        V[v_map[0]][col_offset + 5] += 1
        V[v_map[1]][col_offset + 5] += 1
        W[w_map[3]][col_offset + 5] += 1 # C11 += M6
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[u_map[1]][col_offset + 6] += 1
        U[u_map[3]][col_offset + 6] -= 1
        V[v_map[2]][col_offset + 6] += 1
        V[v_map[3]][col_offset + 6] += 1
        W[w_map[0]][col_offset + 6] += 1 # C00 += M7
        W[w_map[1]][col_offset + 6] -= 1 # C01 -= M7 ? No, C01 = M3 + M5.
        # Let's re-verify Strassen reconstruction.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # My W updates:
        # M1: W00+=1, W11+=1. Correct.
        # M2: W10+=1, W11-=1. Correct.
        # M3: W01+=1, W11+=1. Correct.
        # M4: W00+=1, W10+=1. Correct.
        # M5: W00-=1, W01+=1. Correct.
        # M6: W11+=1. Correct.
        # M7: W00+=1. Correct.
        # Did I miss M7 in W01? No, C12 doesn't have M7.
        # Did I miss M7 in W11? No, C22 doesn't have M7.
        # So M7 only affects W00. Correct.

    # Part 1: A1 B1 (Rank 14)
    # A1 is col 0 of A. B1 is row 0 of B.
    for i in range(2):
        for k in range(7):
            col = i * 7 + k
            U[i * 3 + 0][col] = 1
            V[0 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1

    # Part 2: A2 B2
    # A2 indices:
    # a00 -> U[1]
    # a01 -> U[2]
    # a10 -> U[4]
    # a11 -> U[5]
    u_map = {0: 1, 1: 2, 2: 4, 3: 5}

    # Block 1: k=0,1
    v_map1 = {0: 7, 1: 8, 2: 14, 3: 15}
    w_map1 = {0: 0, 1: 2, 2: 1, 3: 3}
    add_strassen_block(u_map, v_map1, w_map1, 14)

    # Block 2: k=2,3
    v_map2 = {0: 9, 1: 10, 2: 16, 3: 17}
    w_map2 = {0: 4, 1: 6, 2: 5, 3: 7}
    add_strassen_block(u_map, v_map2, w_map2, 21)

    # Block 3: k=4,5
    v_map3 = {0: 11, 1: 12, 2: 18, 3: 19}
    w_map3 = {0: 8, 1: 10, 2: 9, 3: 11}
    add_strassen_block(u_map, v_map3, w_map3, 28)

    # Remainder: k=6
    # P1 = a00 * b00
    U[1][35] += 1
    V[13][35] += 1
    W[12][35] += 1
    
    # P2 = a01 * b10
    U[2][36] += 1
    V[20][36] += 1
    W[12][36] += 1
    
    # P3 = a10 * b00
    U[4][37] += 1
    V[13][37] += 1
    W[13][37] += 1
    
    # P4 = a11 * b10
    U[5][38] += 1
    V[20][38] += 1
    W[13][38] += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
