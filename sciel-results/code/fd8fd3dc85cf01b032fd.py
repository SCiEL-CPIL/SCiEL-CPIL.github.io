"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition using 3 blocks of 2x3x2 (rank 11 each)
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_rows: list of (row, val)
    # v_rows: list of (row, val)
    # w_rows: list of (row, val)
    def add_term(col, u_rows, v_rows, w_rows):
        for r, v in u_rows:
            U[r][col] += v
        for r, v in v_rows:
            V[r][col] += v
        for r, v in w_rows:
            W[r][col] += v
            
    for b in range(3):
        base_col = b * 11
        
        # V row indices for this block
        # B is 3x6. Block b uses cols 2b, 2b+1.
        # V row = j * 6 + k
        v_11 = 0 * 6 + 2 * b
        v_12 = 0 * 6 + 2 * b + 1
        v_21 = 1 * 6 + 2 * b
        v_22 = 1 * 6 + 2 * b + 1
        v_31 = 2 * 6 + 2 * b
        v_32 = 2 * 6 + 2 * b + 1
        
        # W row indices for this block
        # W row = k * 2 + i. C cols 2b, 2b+1.
        # C[0, 2b] -> row (2b)*2 + 0 = 4b
        # C[1, 2b] -> row (2b)*2 + 1 = 4b+1
        # C[0, 2b+1] -> row (2b+1)*2 + 0 = 4b+2
        # C[1, 2b+1] -> row (2b+1)*2 + 1 = 4b+3
        w_00 = 4 * b
        w_10 = 4 * b + 1
        w_01 = 4 * b + 2
        w_11 = 4 * b + 3
        
        # U rows are fixed:
        # a11=0, a12=1, a13=2, a21=3, a22=4, a23=5
        
        # 1. M1 = (a11 + a22)(b11 + b22)
        add_term(base_col + 0, 
                 [(0, 1), (4, 1)], 
                 [(v_11, 1), (v_22, 1)], 
                 [(w_00, 1), (w_11, 1)])
                 
        # 2. M2 = (a21 + a22) b11
        add_term(base_col + 1, 
                 [(3, 1), (4, 1)], 
                 [(v_11, 1)], 
                 [(w_10, 1), (w_11, -1)])
                 
        # 3. M3 = a11 (b12 - b22)
        add_term(base_col + 2, 
                 [(0, 1)], 
                 [(v_12, 1), (v_22, -1)], 
                 [(w_01, 1), (w_11, 1)])
                 
        # 4. M4 = a22 (b21 - b11)
        add_term(base_col + 3, 
                 [(4, 1)], 
                 [(v_21, 1), (v_11, -1)], 
                 [(w_00, 1), (w_10, 1)])
                 
        # 5. M5 = (a11 + a12) b22
        add_term(base_col + 4, 
                 [(0, 1), (1, 1)], 
                 [(v_22, 1)], 
                 [(w_00, -1), (w_01, 1)])
                 
        # 6. M6 = (a21 - a11)(b11 + b12)
        add_term(base_col + 5, 
                 [(3, 1), (0, -1)], 
                 [(v_11, 1), (v_12, 1)], 
                 [(w_11, 1)])
                 
        # 7. M7 = (a12 - a22)(b21 + b22)
        add_term(base_col + 6, 
                 [(1, 1), (4, -1)], 
                 [(v_21, 1), (v_22, 1)], 
                 [(w_00, 1), (w_11, 1)])
                 
        # 8. M8 = a13 b31
        add_term(base_col + 7, 
                 [(2, 1)], 
                 [(v_31, 1)], 
                 [(w_00, 1)])
                 
        # 9. M9 = a13 b32
        add_term(base_col + 8, 
                 [(2, 1)], 
                 [(v_32, 1)], 
                 [(w_01, 1)])
                 
        # 10. M10 = a23 b31
        add_term(base_col + 9, 
                 [(5, 1)], 
                 [(v_31, 1)], 
                 [(w_10, 1)])
                 
        # 11. M11 = a23 b32
        add_term(base_col + 10, 
                 [(5, 1)], 
                 [(v_32, 1)], 
                 [(w_11, 1)])
                 
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
