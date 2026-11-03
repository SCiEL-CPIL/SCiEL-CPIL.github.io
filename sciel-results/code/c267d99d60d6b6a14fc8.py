import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition: 8 independent 2x2x2 Strassen blocks
    # A (2x4) split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # B (4x8): top 2 rows split into 4 blocks of 2x2, bottom 2 rows split into 4 blocks of 2x2
    # For each column-pair s (0..3): C[:, 2s:2s+2] = A_left * B_top[:, 2s:2s+2] + A_right * B_bot[:, 2s:2s+2]
    # Each 2x2*2x2 uses Strassen rank 7. Total: 8 * 7 = 56.
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2: C = X * Y
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)(y00)
    # M3 = (x00)(y01-y11)
    # M4 = (x11)(y10-y00)
    # M5 = (x00+x01)(y11)
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # c00 = M1 - M5 + M4 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 + M3 - M2 + M6
    
    for s in range(4):  # column pair index in output (and in B)
        for t in range(2):  # 0 = top rows of B, 1 = bottom rows of B
            base_r = (s * 2 + t) * 7
            
            # A block columns
            a_col0 = 2 * t
            a_col1 = 2 * t + 1
            # B block rows
            b_row0 = 2 * t
            b_row1 = 2 * t + 1
            # Output columns
            c_col0 = 2 * s
            c_col1 = 2 * s + 1
            
            # U indices: A is 2x4, flattened row-major: row0: 0,1,2,3; row1: 4,5,6,7
            u_x00 = 0 * 4 + a_col0  # A[0, a_col0]
            u_x01 = 0 * 4 + a_col1  # A[0, a_col1]
            u_x10 = 1 * 4 + a_col0  # A[1, a_col0]
            u_x11 = 1 * 4 + a_col1  # A[1, a_col1]
            
            # V indices: B is 4x8, flattened row-major: row r, col c -> r*8 + c
            v_y00 = b_row0 * 8 + c_col0  # B[b_row0, c_col0]
            v_y01 = b_row0 * 8 + c_col1  # B[b_row0, c_col1]
            v_y10 = b_row1 * 8 + c_col0  # B[b_row1, c_col0]
            v_y11 = b_row1 * 8 + c_col1  # B[b_row1, c_col1]
            
            # W indices: C is 2x8, flattened row-major (trace convention W[k*n+i, r] means C[i, k] -> W[k*2+i, r])
            # C[0, c_col0] -> W[c_col0 * 2 + 0, r]
            # C[0, c_col1] -> W[c_col1 * 2 + 0, r]
            # C[1, c_col0] -> W[c_col0 * 2 + 1, r]
            # C[1, c_col1] -> W[c_col1 * 2 + 1, r]
            w_c00 = c_col0 * 2 + 0  # C[0, c_col0]
            w_c01 = c_col1 * 2 + 0  # C[0, c_col1]
            w_c10 = c_col0 * 2 + 1  # C[1, c_col0]
            w_c11 = c_col1 * 2 + 1  # C[1, c_col1]
            
            # M1 = (x00+x11)(y00+y11)
            r = base_r
            U[u_x00][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            V[v_y11][r] += 1
            W[w_c00][r] += 1
            W[w_c11][r] += 1
            
            # M2 = (x10+x11)(y00)
            r = base_r + 1
            U[u_x10][r] += 1
            U[u_x11][r] += 1
            V[v_y00][r] += 1
            W[w_c10][r] += 1
            W[w_c11][r] += -1
            
            # M3 = (x00)(y01-y11)
            r = base_r + 2
            U[u_x00][r] += 1
            V[v_y01][r] += 1
            V[v_y11][r] += -1
            W[w_c01][r] += 1
            W[w_c11][r] += 1
            
            # M4 = (x11)(y10-y00)
            r = base_r + 3
            U[u_x11][r] += 1
            V[v_y10][r] += 1
            V[v_y00][r] += -1
            W[w_c00][r] += 1
            W[w_c10][r] += 1
            
            # M5 = (x00+x01)(y11)
            r = base_r + 4
            U[u_x00][r] += 1
            U[u_x01][r] += 1
            V[v_y11][r] += 1
            W[w_c00][r] += -1
            W[w_c01][r] += 1
            
            # M6 = (x10-x00)(y00+y01)
            r = base_r + 5
            U[u_x10][r] += 1
            U[u_x00][r] += -1
            V[v_y00][r] += 1
            V[v_y01][r] += 1
            W[w_c11][r] += 1
            
            # M7 = (x01-x11)(y10+y11)
            r = base_r + 6
            U[u_x01][r] += 1
            U[u_x11][r] += -1
            V[v_y10][r] += 1
            V[v_y11][r] += 1
            W[w_c00][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
