import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # We use a block-Strassen approach.
    # Split A (2x4) into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    # Split B (4x8) into four 2x2 blocks for each row pair:
    # B_top (rows 0,1) split into 4 col blocks of width 2.
    # B_bot (rows 2,3) split into 4 col blocks of width 2.
    # C (2x8) is sum of A_left * B_top_block + A_right * B_bot_block for each col block.
    
    # There are 4 col blocks and 2 row-blocks (top/bot of B, corresponding to left/right of A).
    # Total 8 independent 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen components for 2x2 matrices X, Y:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2 X:
    # x00 -> A[0, col0]
    # x01 -> A[0, col1]
    # x10 -> A[1, col0]
    # x11 -> A[1, col1]
    
    # Local indices for 2x2 Y:
    # y00 -> B[row0, col0]
    # y01 -> B[row0, col1]
    # y10 -> B[row1, col0]
    # y11 -> B[row1, col1]
    
    # Local indices for 2x2 Z:
    # z00 -> C[0, col0]
    # z01 -> C[0, col1]
    # z10 -> C[1, col0]
    # z11 -> C[1, col1]
    
    for s in range(4):  # Column block index 0..3
        for t in range(2):  # Row block index 0..1 (t=0: A_left, B_top; t=1: A_right, B_bot)
            base_r = (s * 2 + t) * 7
            
            # A columns: 2t, 2t+1
            a_col0 = 2 * t
            a_col1 = 2 * t + 1
            
            # B rows: 2t, 2t+1
            b_row0 = 2 * t
            b_row1 = 2 * t + 1
            
            # C columns: 2s, 2s+1
            c_col0 = 2 * s
            c_col1 = 2 * s + 1
            
            # U indices (A)
            u_x00 = 0 * 4 + a_col0
            u_x01 = 0 * 4 + a_col1
            u_x10 = 1 * 4 + a_col0
            u_x11 = 1 * 4 + a_col1
            
            # V indices (B)
            v_y00 = b_row0 * 8 + c_col0
            v_y01 = b_row0 * 8 + c_col1
            v_y10 = b_row1 * 8 + c_col0
            v_y11 = b_row1 * 8 + c_col1
            
            # W indices (C) - Note: W[k*n + i]
            w_z00 = c_col0 * 2 + 0
            w_z01 = c_col1 * 2 + 0
            w_z10 = c_col0 * 2 + 1
            w_z11 = c_col1 * 2 + 1
            
            # M1
            r = base_r + 0
            U[u_x00][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += 1
            W[w_z11][r] += 1
            
            # M2
            r = base_r + 1
            U[u_x10][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            W[w_z10][r] += 1
            W[w_z11][r] += -1
            
            # M3
            r = base_r + 2
            U[u_x00][r] += 1
            V[v_y01][r] += 1
            V[v_y11][r] += -1
            W[w_z01][r] += 1
            W[w_z11][r] += 1
            
            # M4
            r = base_r + 3
            U[u_x11][r] += 1
            V[v_y10][r] += 1
            V[v_y00][r] += -1
            W[w_z00][r] += 1
            W[w_z10][r] += 1
            
            # M5
            r = base_r + 4
            U[u_x00][r] += 1
            U[u_x01][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += -1
            W[w_z01][r] += 1
            
            # M6
            r = base_r + 5
            U[u_x10][r] += 1
            U[u_x00][r] += -1
            V[v_y00][r] += 1
            V[v_y01][r] += 1
            W[w_z11][r] += 1
            
            # M7
            r = base_r + 6
            U[u_x01][r] += 1
            U[u_x11][r] += -1
            V[v_y10][r] += 1
            V[v_y11][r] += 1
            W[w_z00][r] += 1

    if time.time() - start_time > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
