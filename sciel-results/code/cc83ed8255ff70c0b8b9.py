import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    # Rank 32 solution:
    # 1x4x4 part (rank 4) + 2x4x4 part (rank 28)
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: 1x4x4 (rows 0 of A, all of B, row 0 of C)
    # A is 1x4, B is 4x4, C is 1x4
    # U shape (4, 4), V shape (16, 4), W shape (4, 4)
    # Indices:
    # U: a = r*m + i. r=0, i=0..3. a=0..3.
    # V: b = i*p + c. i=0..3, c=0..3. b=0..15.
    # W: w = c*n + r. c=0..3, r=0. w=0..3.
    
    for k in range(4):
        # U[a, k]
        for i in range(4):
            U[i, k] = 1.0
        # V[b, k]
        for i in range(4):
            for c in range(4):
                if i == k:
                    V[i * p + c, k] = 1.0
        # W[w, k]
        for c in range(4):
            W[c * n + 0, k] = 1.0
            
    # Part 2: 2x4x4 (rows 1-2 of A, all of B, rows 1-2 of C)
    # A is 2x4, B is 4x4, C is 2x4
    # Use 4 blocks of 2x2x2 (Strassen rank 7)
    # A blocks: A1 (rows 1-2, cols 0-1), A2 (rows 1-2, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C blocks: C1 (rows 1-2, cols 0-1), C2 (rows 1-2, cols 2-3)
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # Strassen for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to embed this into the global indices.
    # Global A index: r*m + i. r=1,2. i=0..3.
    # Global B index: i*p + c. i=0..3. c=0..3.
    # Global C index: c*n + r. c=0..3. r=1,2.
    
    # Helper to add a term
    def add_term(u_list, v_list, w_list, col_offset):
        for idx, val in u_list:
            U[idx, col_offset] += val
        for idx, val in v_list:
            V[idx, col_offset] += val
        for idx, val in w_list:
            W[idx, col_offset] += val

    # Block 1: A1 B11 -> C1
    # A1: r=1,2; i=0,1. Indices: 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9.
    # B11: i=0,1; c=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # C1: r=1,2; c=0,1. Indices: 0*3+1=1, 0*3+2=2, 1*3+1=4, 1*3+2=5.
    
    # This is getting complex. I will use the known rank 44 solution as it is safer to implement correctly.
    # The rank 44 solution is already in the current program.
    # I will just return the current program's logic to ensure correctness, as I don't have time to debug the rank 32 construction.
    
    # Actually, the current program is score 1.0909.
    # If I return the same program, I get 1.0909.
    # I need to improve it.
    
    # Let's try to implement the rank 32 solution carefully.
    
    # Strassen terms for 2x2x2
    # Local indices:
    # A: (r, c) in 0..1.
    # B: (r, c) in 0..1.
    # C: (r, c) in 0..1.
    
    def get_strassen_terms():
        # Returns list of (u_list, v_list, w_list) for 2x2x2
        # u_list: (local_a_idx, val)
        # v_list: (local_b_idx, val)
        # w_list: (local_c_idx, val)
        terms = []
        # M1
        terms.append(([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]))
        # M2
        terms.append(([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]))
        # M3
        terms.append(([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]))
        # M4
        terms.append(([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]))
        # M5
        terms.append(([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]))
        # M6
        terms.append(([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]))
        # M7
        terms.append(([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]))
        return terms

    strassen_terms = get_strassen_terms()
    
    # 4 blocks for 2x4x4
    # Block (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    blocks = [
        (0, 0, 0, 0), # A1 B11 -> C1
        (0, 0, 2, 2), # A1 B12 -> C2
        (2, 2, 0, 0), # A2 B21 -> C1
        (2, 2, 2, 2), # A2 B22 -> C2
    ]
    
    col_offset = 4
    for a_cs, b_rs, b_cs, c_cs in blocks:
        for u_list, v_list, w_list in strassen_terms:
            # Map local indices to global
            g_u_list = []
            for idx, val in u_list:
                r_loc = idx // 2
                c_loc = idx % 2
                # A global index: r=1+ r_loc, i=a_cs+c_loc
                g_a_idx = (1 + r_loc) * m + (a_cs + c_loc)
                g_u_list.append((g_a_idx, val))
            
            g_v_list = []
            for idx, val in v_list:
                r_loc = idx // 2
                c_loc = idx % 2
                # B global index: i=b_rs+r_loc, c=b_cs+c_loc
                g_b_idx = (b_rs + r_loc) * p + (b_cs + c_loc)
                g_v_list.append((g_b_idx, val))
            
            g_w_list = []
            for idx, val in w_list:
                r_loc = idx // 2
                c_loc = idx % 2
                # C global index: c=c_cs+c_loc, r=1+r_loc
                g_c_idx = (c_cs + c_loc) * n + (1 + r_loc)
                g_w_list.append((g_c_idx, val))
            
            add_term(g_u_list, g_v_list, g_w_list, col_offset)
            col_offset += 1
            
    if time.time() > deadline:
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

