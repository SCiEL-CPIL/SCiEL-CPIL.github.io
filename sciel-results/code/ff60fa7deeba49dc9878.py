import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # Total rank
    rank = 50
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices for 2x2x2:
    # U row i*2+j
    # V row j*2+k
    # W row k*2+i
    
    U_strassen = [
        [1, 0, 1, 1, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    
    W_strassen = [
        [1, 0, 0, -1, 0, -1, 1], # C00 (k=0, i=0) -> idx 0
        [0, 1, 0, 1, 0, 0, 0], # C10 (k=0, i=1) -> idx 1
        [0, 0, 1, 0, 1, 0, 0], # C01 (k=1, i=0) -> idx 2
        [1, 0, -1, 0, 0, 1, -1] # C11 (k=1, i=1) -> idx 3
    ]
    
    # Schoolbook decomposition for 2x2x1
    # U: 4x4, V: 2x4, W: 2x4
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    def add_block(u_offset, v_offset, w_offset, rank_offset):
        # u_offset: starting row index in U for the 2x2 A block
        # v_offset: starting row index in V for the 2x7 B block
        # w_offset: starting row index in W for the 2x7 C block
        # rank_offset: starting column index in U, V, W
        
        # B is 2x7. Split into 3 pairs (cols 0-1, 2-3, 4-5) and 1 single (col 6)
        
        # Pair 0: cols 0, 1
        # V rows for B pair 0: v_offset + 0*7 + 0, v_offset + 0*7 + 1 (j=0)
        #                   v_offset + 1*7 + 0, v_offset + 1*7 + 1 (j=1)
        # Local V indices 0..3 map to:
        # 0 -> v_offset
        # 1 -> v_offset + 1
        # 2 -> v_offset + 7
        # 3 -> v_offset + 8
        
        for term in range(7):
            col = rank_offset + term
            # U
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_offset + li][col] += U_strassen[li][term]
            
            # V
            # Map local V index to global
            # local 0 -> global v_offset + 0
            # local 1 -> global v_offset + 1
            # local 2 -> global v_offset + 7
            # local 3 -> global v_offset + 8
            v_map = [0, 1, 7, 8]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            # W
            # Local W index 0..3 maps to w_offset + 0..3
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Pair 1: cols 2, 3
        # V rows: v_offset + 2, v_offset + 3, v_offset + 9, v_offset + 10
        rank_offset += 7
        for term in range(7):
            col = rank_offset + term
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_offset + li][col] += U_strassen[li][term]
            
            v_map = [2, 3, 9, 10]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Pair 2: cols 4, 5
        # V rows: v_offset + 4, v_offset + 5, v_offset + 11, v_offset + 12
        rank_offset += 7
        for term in range(7):
            col = rank_offset + term
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_offset + li][col] += U_strassen[li][term]
            
            v_map = [4, 5, 11, 12]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Single: col 6
        # V rows: v_offset + 6, v_offset + 13
        rank_offset += 7
        for term in range(4):
            col = rank_offset + term
            for li in range(4):
                if U_school[li][term] != 0:
                    U[u_offset + li][col] += U_school[li][term]
            
            v_map = [6, 13]
            for li in range(2):
                if V_school[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_school[li][term]
            
            for li in range(2):
                if W_school[li][term] != 0:
                    W[w_offset + li][col] += W_school[li][term]
                    
        return rank_offset + 4 # Returns next rank offset

    # Block A: A_L (cols 0,1), B_T (rows 0,1)
    # U rows for A_L:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # We need to map local U indices 0..3 to these global rows.
    # Local 0 (A00) -> Global 0
    # Local 1 (A01) -> Global 1
    # Local 2 (A10) -> Global 4
    # Local 3 (A11) -> Global 5
    
    # Instead of hardcoding in add_block, we can modify U access.
    # But add_block uses u_offset + li.
    # So we can't just use a single offset.
    
    # Let's refactor add_block to take a list of U row indices.
    
    def add_block_v2(u_rows, v_offset, w_offset, rank_offset):
        # u_rows: list of 4 global row indices in U corresponding to local 0,1,2,3
        
        # Pair 0
        for term in range(7):
            col = rank_offset + term
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_rows[li]][col] += U_strassen[li][term]
            
            v_map = [0, 1, 7, 8]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Pair 1
        rank_offset += 7
        for term in range(7):
            col = rank_offset + term
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_rows[li]][col] += U_strassen[li][term]
            
            v_map = [2, 3, 9, 10]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Pair 2
        rank_offset += 7
        for term in range(7):
            col = rank_offset + term
            for li in range(4):
                if U_strassen[li][term] != 0:
                    U[u_rows[li]][col] += U_strassen[li][term]
            
            v_map = [4, 5, 11, 12]
            for li in range(4):
                if V_strassen[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_strassen[li][term]
            
            for li in range(4):
                if W_strassen[li][term] != 0:
                    W[w_offset + li][col] += W_strassen[li][term]
                    
        # Single
        rank_offset += 7
        for term in range(4):
            col = rank_offset + term
            for li in range(4):
                if U_school[li][term] != 0:
                    U[u_rows[li]][col] += U_school[li][term]
            
            v_map = [6, 13]
            for li in range(2):
                if V_school[li][term] != 0:
                    V[v_offset + v_map[li]][col] += V_school[li][term]
            
            for li in range(2):
                if W_school[li][term] != 0:
                    W[w_offset + li][col] += W_school[li][term]
                    
        return rank_offset + 4

    # Block A
    u_rows_A = [0, 1, 4, 5]
    v_offset_A = 0
    w_offset_A = 0
    next_rank = add_block_v2(u_rows_A, v_offset_A, w_offset_A, 0)
    
    # Block B
    # A_R (cols 2,3)
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    u_rows_B = [2, 3, 6, 7]
    v_offset_B = 14 # Rows 2,3 of B start at 2*7 = 14
    w_offset_B = 0 # Same C
    next_rank = add_block_v2(u_rows_B, v_offset_B, w_offset_B, next_rank)
    
    return {"U": U, "V": V, "W": W}

