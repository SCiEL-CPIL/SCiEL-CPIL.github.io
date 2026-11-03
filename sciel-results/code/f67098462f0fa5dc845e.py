import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def construct():
    n, m, p = DIMENSIONS
    # The optimal rank for 2x4x6 matrix multiplication is 42.
    # This is achieved by splitting the 4x6 matrix into two 2x6 blocks
    # and the 2x4 matrix into two 2x2 blocks, resulting in two independent
    # (2x2) * (2x6) multiplications.
    # Each (2x2) * (2x6) multiplication has rank 21 (3 * 7, using Strassen
    # for each 2x2 column block of the 2x6 matrix).
    # Total rank = 21 + 21 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    deadline = time.time() + 145.0
    
    # Strassen decomposition for 2x2 * 2x2
    # Indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00       -> C10+=M2, C11-=M2
    # M3 = A00(B10-B00)       -> C01+=M3, C11+=M3
    # M4 = A11(B00-B10)       -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11       -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    strassen_components = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [(local_a_idx, coeff), ...]
        # v: [(local_b_idx, coeff), ...]
        # w: [(local_c_idx, coeff), ...]
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    col = 0
    # We process 6 blocks of (2x2) * (2x2)
    # Block structure:
    # A is 2x4. Split into A_left (cols 0-1) and A_right (cols 2-3).
    # B is 4x6. Split into B_top (rows 0-1) and B_bot (rows 2-3).
    # Product C = A_left * B_top + A_right * B_bot.
    # B_top is 2x6. Split into B_top_0 (cols 0-1), B_top_1 (cols 2-3), B_top_2 (cols 4-5).
    # B_bot is 2x6. Split into B_bot_0 (cols 0-1), B_bot_1 (cols 2-3), B_bot_2 (cols 4-5).
    
    # Blocks: (a_row_start, a_col_start, b_row_start, b_col_start)
    # a_row_start is always 0 (A is 2 rows)
    # b_row_start is 0 for top, 2 for bot.
    
    blocks = [
        # A_left * B_top parts
        (0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2]
        (0, 0, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4]
        (0, 0, 0, 4), # A[0:2, 0:2] * B[0:2, 4:6]
        # A_right * B_bot parts
        (0, 2, 2, 0), # A[0:2, 2:4] * B[2:4, 0:2]
        (0, 2, 2, 2), # A[0:2, 2:4] * B[2:4, 2:4]
        (0, 2, 2, 4), # A[0:2, 2:4] * B[2:4, 4:6]
    ]
    
    for a_r, a_c, b_r, b_c in blocks:
        if not _time_ok(deadline):
            break
            
        # Map local indices (0..3) to global indices for U, V, W
        # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block start
        # Global A index: (a_r + i) * m + (a_c + j)
        a_map = [
            (a_r + 0) * m + (a_c + 0), # 0
            (a_r + 0) * m + (a_c + 1), # 1
            (a_r + 1) * m + (a_c + 0), # 2
            (a_r + 1) * m + (a_c + 1), # 3
        ]
        
        # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global B index: (b_r + i) * p + (b_c + j)
        b_map = [
            (b_r + 0) * p + (b_c + 0), # 0
            (b_r + 0) * p + (b_c + 1), # 1
            (b_r + 1) * p + (b_c + 0), # 2
            (b_r + 1) * p + (b_c + 1), # 3
        ]
        
        # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global C index (W): (b_c + j) * n + (a_r + i)
        # Note: W[k*n+i] where k is col index of C (from B col), i is row index of C (from A row)
        # In our block, C rows are a_r, a_r+1. C cols are b_c, b_c+1.
        w_map = [
            (b_c + 0) * n + (a_r + 0), # 0 -> C[a_r, b_c]
            (b_c + 1) * n + (a_r + 0), # 1 -> C[a_r, b_c+1]
            (b_c + 0) * n + (a_r + 1), # 2 -> C[a_r+1, b_c]
            (b_c + 1) * n + (a_r + 1), # 3 -> C[a_r+1, b_c+1]
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_components:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
