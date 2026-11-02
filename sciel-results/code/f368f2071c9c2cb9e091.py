import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use 4 blocks of 2x2 Strassen (rank 7 each) = rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2x2 decomposition
    # For X (2x2) * Y (2x2) = Z (2x2):
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z00 = x00*y00 + x10*y10
    # Z01 = x00*y01 + x10*y11
    # Z10 = x01*y00 + x11*y10
    # Z11 = x01*y01 + x11*y11
    #
    # Strassen products (7):
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y10)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    #
    # Z00 = M1 - M4 + M6 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M5 - M6
    
    # For each Strassen product s, we need:
    # U factors: which (i, j) positions in X contribute, and with what coefficient
    # V factors: which (j, k) positions in Y contribute, and with what coefficient
    # W factors: which (k, i) positions in Z are produced, and with what coefficient
    
    # Define Strassen terms: (list of (x_idx, coeff)), (list of (y_idx, coeff)), (list of (z_idx, coeff))
    # x_idx: 0=x00, 1=x01, 2=x10, 3=x11
    # y_idx: 0=y00, 1=y01, 2=y10, 3=y11
    # z_idx: 0=Z00, 1=Z01, 2=Z10, 3=Z11
    
    strassen_terms = [
        # M1 = (x00+x11)*(y00+y11), contributes to Z00 (+1), Z11 (+1)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (x10+x11)*y00, contributes to Z10 (+1)
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3 = x00*(y01-y10), contributes to Z01 (+1), Z11 (-1)
        ([(0, 1)], [(1, 1), (2, -1)], [(1, 1), (3, -1)]),
        # M4 = x11*(y10-y00), contributes to Z00 (-1), Z10 (+1)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]),
        # M5 = (x00+x01)*y11, contributes to Z01 (+1), Z11 (+1)
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # M6 = (x10-x00)*(y00+y10), contributes to Z00 (+1), Z11 (-1)
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(0, 1), (3, -1)]),
        # M7 = (x01-x11)*(y01+y11), contributes to Z00 (+1), Z11 (-1)
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)], [(0, 1), (3, -1)]),
    ]
    
    column = 0
    for block in range(2):  # 2 blocks of 2 columns
        for s in range(7):  # 7 Strassen products per block
            x_factors, y_factors, z_factors = strassen_terms[s]
            
            for (xi, x_coeff) in x_factors:
                # xi: 0=x00, 1=x01, 2=x10, 3=x11
                xi_row = xi // 2  # 0 or 1 (row of X)
                xi_col = xi % 2   # 0 or 1 (col of X within block)
                a_row = xi_row
                a_col = block * 2 + xi_col
                U[a_row * m + a_col][column] += x_coeff
            
            for (yi, y_coeff) in y_factors:
                # yi: 0=y00, 1=y01, 2=y10, 3=y11
                yi_row = yi // 2   # 0 or 1 (row of Y within block)
                yi_col = yi % 2    # 0 or 1 (col of Y)
                b_row = block * 2 + yi_row
                b_col = yi_col
                V[b_row * p + b_col][column] += y_coeff
            
            for (zi, z_coeff) in z_factors:
                # zi: 0=Z00, 1=Z01, 2=Z10, 3=Z11
                zi_row = zi // 2   # 0 or 1 (row of Z)
                zi_col = zi % 2    # 0 or 1 (col of Z within block)
                c_row = zi_row
                c_col = block * 2 + zi_col
                W[c_col * n + c_row][column] += z_coeff
            
            column += 1
    
    return {"U": U, "V": V, "W": W}
