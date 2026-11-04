"""Rewritten program using a Strassen-based block decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Global dimensions
    num_u = n * m  # 8
    num_v = m * p  # 16
    num_w = p * n  # 8
    
    rank = 28
    U = [[0] * rank for _ in range(num_u)]
    V = [[0] * rank for _ in range(num_v)]
    W = [[0] * rank for _ in range(num_w)]
    
    col = 0
    
    # Strassen 2x2 terms (local indices 0..3)
    # u, v, w are lists of 4 floats
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        for u_loc, v_loc, w_loc in strassen_terms:
            # Map local indices to global
            # Local x_{r,c} -> A[a_row_off+r, a_col_off+c]
            # Global U index: (a_row_off+r)*m + (a_col_off+c)
            # Local y_{r,c} -> B[b_row_off+r, b_col_off+c]
            # Global V index: (b_row_off+r)*p + (b_col_off+c)
            # Local z_{r,c} -> C[c_row_off+r, c_col_off+c]
            # Global W index: (c_col_off+c)*n + (c_row_off+r)
            
            for l in range(4):
                # Unpack local index l to r, c
                # l=0: (0,0), l=1: (0,1), l=2: (1,0), l=3: (1,1)
                if l == 0:
                    r, c = 0, 0
                elif l == 1:
                    r, c = 0, 1
                elif l == 2:
                    r, c = 1, 0
                elif l == 3:
                    r, c = 1, 1
                
                coeff_u = u_loc[l]
                coeff_v = v_loc[l]
                coeff_w = w_loc[l]
                
                if coeff_u != 0:
                    u_idx = (a_row_off + r) * m + (a_col_off + c)
                    U[u_idx][col] += coeff_u
                
                if coeff_v != 0:
                    v_idx = (b_row_off + r) * p + (b_col_off + c)
                    V[v_idx][col] += coeff_v
                
                if coeff_w != 0:
                    w_idx = (c_col_off + c) * n + (c_row_off + r)
                    W[w_idx][col] += coeff_w
            
            col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_block(0, 2, 2, 0, 0, 0)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_block(0, 0, 0, 2, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_block(0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
