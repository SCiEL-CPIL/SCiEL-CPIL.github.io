"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Rank is 56 using Block-Strassen
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # 8 sub-problems:
    # 0-3: A_L (cols 0,1) * B_T blocks (rows 0,1; cols 0-1, 2-3, 4-5, 6-7)
    # 4-7: A_R (cols 2,3) * B_B blocks (rows 2,3; cols 0-1, 2-3, 4-5, 6-7)
    
    for s in range(8):
        if s < 4:
            a_col_offset = 0
            b_row_offset = 0
            b_col_offset = 2 * s
        else:
            a_col_offset = 2
            b_row_offset = 2
            b_col_offset = 2 * (s - 4)
        
        base_rank = s * 7
        
        def add_strassen_term(term_idx, u_terms, v_terms, w_terms):
            col = base_rank + term_idx
            for i, j, coeff in u_terms:
                U[i * m + j][col] += coeff
            for j, k, coeff in v_terms:
                V[j * p + k][col] += coeff
            for k, i, coeff in w_terms:
                W[k * n + i][col] += coeff

        # M1 = (X11+X22)(Y11+Y22)
        add_strassen_term(0, 
            [(0, a_col_offset, 1), (1, a_col_offset+1, 1)],
            [(b_row_offset, b_col_offset, 1), (b_row_offset+1, b_col_offset+1, 1)],
            [(b_col_offset, 0, 1), (b_col_offset+1, 1, 1)]
        )
        
        # M2 = (X21+X22)Y11
        add_strassen_term(1,
            [(1, a_col_offset, 1), (1, a_col_offset+1, 1)],
            [(b_row_offset, b_col_offset, 1)],
            [(b_col_offset, 1, 1)]
        )
        
        # M3 = X11(Y12-Y22)
        add_strassen_term(2,
            [(0, a_col_offset, 1)],
            [(b_row_offset, b_col_offset+1, 1), (b_row_offset+1, b_col_offset+1, -1)],
            [(b_col_offset+1, 0, 1), (b_col_offset+1, 1, 1)]
        )
        
        # M4 = X22(Y21-Y11)
        add_strassen_term(3,
            [(1, a_col_offset+1, 1)],
            [(b_row_offset+1, b_col_offset, 1), (b_row_offset, b_col_offset, -1)],
            [(b_col_offset, 0, 1), (b_col_offset, 1, 1)]
        )
        
        # M5 = (X11+X12)Y22
        add_strassen_term(4,
            [(0, a_col_offset, 1), (0, a_col_offset+1, 1)],
            [(b_row_offset+1, b_col_offset+1, 1)],
            [(b_col_offset, 0, -1), (b_col_offset+1, 0, 1)]
        )
        
        # M6 = (X21-X11)(Y11+Y12)
        add_strassen_term(5,
            [(1, a_col_offset, 1), (0, a_col_offset, -1)],
            [(b_row_offset, b_col_offset, 1), (b_row_offset, b_col_offset+1, 1)],
            [(b_col_offset+1, 1, 1)]
        )
        
        # M7 = (X12-X22)(Y21+Y22)
        add_strassen_term(6,
            [(0, a_col_offset+1, 1), (1, a_col_offset+1, -1)],
            [(b_row_offset+1, b_col_offset, 1), (b_row_offset+1, b_col_offset+1, 1)],
            [(b_col_offset, 0, 1)]
        )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
