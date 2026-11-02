"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Strassen 7 terms for 2x2 multiplication
    # Each term: (U_coeffs on X, V_coeffs on Y, W_coeffs on Z)
    # X entries indexed 0..3: X00=0, X01=1, X10=2, X11=3
    # Y entries indexed 0..3: Y00=0, Y01=1, Y10=2, Y11=3
    # Z entries indexed 0..3: Z00=0, Z01=1, Z10=2, Z11=3
    
    # Term definitions: (list of (x_idx, coeff), list of (y_idx, coeff), list of (z_idx, coeff))
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (X10+X11)Y00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = X00(Y01-Y11)
        ([(0, 1)], [(1, 1), (3, -1)], [(0, -1), (1, 1)]),
        # M4 = X11(Y10-Y00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (X00+X01)Y11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (X10-X00)(Y00+Y01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (X01-X11)(Y10+Y11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),
    ]
    
    rank = 4 * 7  # 4 products, 7 terms each
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Four products: (x_col_offset, y_row_offset, y_col_offset)
    products = [
        (0, 0, 0),  # A1 * B11
        (0, 0, 2),  # A1 * B12
        (2, 2, 0),  # A2 * B21
        (2, 2, 2),  # A2 * B22
    ]
    
    col = 0
    for (x_col_off, y_row_off, y_col_off) in products:
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            # U: for each (x_idx, coeff), map to A row/col
            for (x_idx, coeff) in u_coeffs:
                x_row = x_idx // 2  # 0 or 1
                x_col = x_idx % 2   # 0 or 1
                a_row = x_row
                a_col = x_col_off + x_col
                U[a_row * m + a_col][col] += coeff
            
            # V: for each (y_idx, coeff), map to B row/col
            for (y_idx, coeff) in v_coeffs:
                y_row = y_idx // 2  # 0 or 1
                y_col = y_idx % 2   # 0 or 1
                b_row = y_row_off + y_row
                b_col = y_col_off + y_col
                V[b_row * p + b_col][col] += coeff
            
            # W: for each (z_idx, coeff), map to C row/col
            for (z_idx, coeff) in w_coeffs:
                z_row = z_idx // 2  # 0 or 1
                z_col = z_idx % 2   # 0 or 1
                c_row = y_row_off + z_row
                c_col = x_col_off + z_col
                # W[k*n + i, r] where k is row of C, i is column of C
                W[c_col * n + c_row][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
