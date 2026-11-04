"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-reduced exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix using Strassen's algorithm.
    
    The decomposition has rank 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # P0 = (X00 + X11)(Y00 + Y11)
    # P1 = (X10 + X11)Y00
    # P2 = X00(Y10 - Y11)
    # P3 = X11(Y01 - Y00)
    # P4 = (X00 + X01)Y11
    # P5 = (X10 - X00)(Y00 + Y01)
    # P6 = (X01 - X11)(Y10 + Y11)
    
    # Coefficients for X (U) and Y (V) for each term t
    # X indices: (i, j) in 0..1
    # Y indices: (j, k) in 0..1
    
    # term: (x_coeffs, y_coeffs)
    # x_coeffs is list of (i, j, coeff)
    # y_coeffs is list of (j, k, coeff)
    
    terms = [
        # P0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # P1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # P2
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # P3
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # P4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # P5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # P6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
    ]
    
    # Coefficients for result M[i,k] from P_t
    # M00 = P0 + P3 - P4 + P6
    # M01 = P2 + P4
    # M10 = P1 + P3
    # M11 = P0 - P1 + P2 + P5
    
    result_coeffs = [
        # i=0, k=0: P0, P3, -P4, P6
        [1, 0, 0, 1, -1, 0, 1],
        # i=0, k=1: P2, P4
        [0, 0, 1, 0, 1, 0, 0],
        # i=1, k=0: P1, P3
        [0, 1, 0, 1, 0, 0, 0],
        # i=1, k=1: P0, -P1, P2, P5
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for S in range(6):
        # Determine offsets
        # S=0,1,2 use A[:, 0:2], S=3,4,5 use A[:, 2:4]
        x_col_off = (S // 3) * 2
        # S=0,3 use B[0:2, :], S=1,4 use B[0:2, 2:4]... 
        # Wait, B row offset:
        # S=0,1,2 -> B rows 0,1. S=3,4,5 -> B rows 2,3.
        y_row_off = (S // 3) * 2
        # B col offset:
        # S=0,3 -> cols 0,1. S=1,4 -> cols 2,3. S=2,5 -> cols 4,5.
        y_col_off = (S % 3) * 2
        
        # Output offset
        c_off = (S % 3) * 2
        
        for t in range(7):
            x_coeffs, y_coeffs = terms[t]
            
            # Fill U
            for (i_loc, j_loc, coeff) in x_coeffs:
                u_row = i_loc
                u_col = x_col_off + j_loc
                U[u_row * m + u_col][col] += coeff
            
            # Fill V
            for (j_loc, k_loc, coeff) in y_coeffs:
                v_row = y_row_off + j_loc
                v_col = y_col_off + k_loc
                V[v_row * p + v_col][col] += coeff
            
            # Fill W
            for i_loc in range(2):
                for k_loc in range(2):
                    coeff = result_coeffs[i_loc * 2 + k_loc][t]
                    if coeff != 0:
                        w_row = c_off + k_loc
                        w_col = i_loc
                        W[w_row * n + w_col][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
