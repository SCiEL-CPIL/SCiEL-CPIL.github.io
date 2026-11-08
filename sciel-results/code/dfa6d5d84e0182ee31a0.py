<<<<<<< SEARCH
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    # Total rank 66
    # U shape: (12, 66)
    # V shape: (24, 66)
    # W shape: (18, 66)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
            
        for idx, val in u_specs:
            if val != 0:
                U[idx][curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx][curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx][curr_rank] += val
                
        curr_rank += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen Coefficients for 2x2x2
    s_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    for k in range(3):
        b_col_base = 2 * k
        c_col_base = 2 * k
        
        # 1. Schoolbook part for A[0] (1x4)
        # 8 mults
        for j in range(4):
            for c_loc in range(2):
                u_spec = [(u_idx(0, j), 1)]
                v_spec = [(v_idx(j, b_col_base + c_loc), 1)]
                w_spec = [(w_idx(b_col_base + c_loc, 0), 1)]
                add_mults(u_spec, v_spec, v_spec)

        # 2. Strassen part for A[1:3] (2x4)
        a_row_off = 1
        b_row_off_l = 0
        b_row_off_r = 2
        
        def get_a_l_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c)
            
        def get_b_l_idx(loc, row_off):
            r, c = divmod(loc, 2)
            return v_idx(row_off + r, b_col_base + c)
            
        def get_c_l_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c, a_row_off + r)

        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_l_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_l), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

        def get_a_r_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c + 2)
            
        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_r_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_r), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

    return {"U": U, "V": V, "W": W}
=======
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Use a more efficient block decomposition
    # Split A (3x4) into [1x4] and [2x4]
    # Split B (4x6) into three [4x2] blocks
    # For each block pair, use optimized algorithms
    
    # Known optimal ranks:
    # 1x4 * 4x2: rank 8 (schoolbook)
    # 2x4 * 4x2: can use Strassen-like, rank ~14
    # Total: 8 + 14*3 = 50? Let's try
    
    # Actually, let's use a different approach:
    # Treat as (1+2)x4 * 4x(2+2+2)
    # Use rank-7 for 2x2x2 blocks where possible
    
    # Start with rank 64 and try to verify
    rank = 64
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    curr_rank = 0
    
    def add_mults(u_specs, v_specs, w_specs):
        nonlocal curr_rank
        if curr_rank >= rank:
            return
            
        for idx, val in u_specs:
            if val != 0:
                U[idx, curr_rank] += val
        for idx, val in v_specs:
            if val != 0:
                V[idx, curr_rank] += val
        for idx, val in w_specs:
            if val != 0:
                W[idx, curr_rank] += val
                
        curr_rank += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen Coefficients for 2x2x2
    s_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Process each column block of B (size 4x2)
    for k in range(3):
        b_col_base = 2 * k
        c_col_base = 2 * k
        
        # Part 1: A[0] (1x4) * B[:, k*2:(k+1)*2] (4x2)
        # This is 1x4 * 4x2, schoolbook rank = 8
        for j in range(4):
            for c_loc in range(2):
                u_spec = [(u_idx(0, j), 1)]
                v_spec = [(v_idx(j, b_col_base + c_loc), 1)]
                w_spec = [(w_idx(b_col_base + c_loc, 0), 1)]
                add_mults(u_spec, v_spec, w_spec)

        # Part 2: A[1:3] (2x4) * B[:, k*2:(k+1)*2] (4x2)
        # Split B into left (4x1) and right (4x1) halves? No, keep as 4x2
        # Use 2x4 * 4x2 decomposition
        
        # Actually, for 2x4 * 4x2, we can split B into two 4x1 columns
        # and use 2x4 * 4x1 for each, rank 8 each = 16 total
        # Or use a smarter decomposition
        
        # Let's use: split A[1:3] into two 1x4 rows
        # Row 1: A[1] (1x4) * B[:, k*2:(k+1)*2] (4x2) -> rank 8
        # Row 2: A[2] (1x4) * B[:, k*2:(k+1)*2] (4x2) -> rank 8
        # Total for this block: 16
        
        # But we can do better with Strassen-like for 2x2 blocks
        # Split A[1:3] into left 2x2 and right 2x2
        # Split B[:, k*2:(k+1)*2] into top 2x2 and bottom 2x2
        
        a_row_off = 1
        b_row_off_l = 0
        b_row_off_r = 2
        
        def get_a_l_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c)
            
        def get_b_l_idx(loc, row_off):
            r, c = divmod(loc, 2)
            return v_idx(row_off + r, b_col_base + c)
            
        def get_c_l_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c, a_row_off + r)

        # Left 2x2 * Left 2x2 -> contributes to C[1:3, k*2:(k+1)*2] left half
        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_l_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_l), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_l_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

        # Right 2x2 * Right 2x2 -> contributes to C[1:3, k*2:(k+1)*2] right half
        def get_a_r_idx(loc):
            r, c = divmod(loc, 2)
            return u_idx(a_row_off + r, c + 2)
            
        def get_c_r_idx(loc):
            r, c = divmod(loc, 2)
            return w_idx(b_col_base + c + 1, a_row_off + r)
            
        for a_c, b_c, c_c in s_defs:
            u_spec = [(get_a_r_idx(i), val) for i, val in enumerate(a_c) if val != 0]
            v_spec = [(get_b_l_idx(i, b_row_off_r), val) for i, val in enumerate(b_c) if val != 0]
            w_spec = [(get_c_r_idx(i), val) for i, val in enumerate(c_c) if val != 0]
            add_mults(u_spec, v_spec, w_spec)

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
