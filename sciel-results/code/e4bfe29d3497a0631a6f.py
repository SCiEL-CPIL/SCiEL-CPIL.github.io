DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen factors for 2x2 multiplication
    strassen_x = [
        [[1, 0], [1, 0]],  # P1
        [[0, 0], [1, 1]],  # P2
        [[1, 0], [0, 0]],  # P3
        [[0, 0], [0, 1]],  # P4
        [[1, 1], [0, 0]],  # P5
        [[0, 0], [1, -1]], # P6
        [[0, 1], [0, -1]], # P7
    ]
    strassen_y = [
        [[1, 1], [1, 1]],  # P1
        [[1, 0], [0, 0]],  # P2
        [[0, 1], [0, -1]], # P3
        [[0, -1], [1, 0]], # P4
        [[0, 0], [0, 1]],  # P5
        [[1, 1], [0, 0]],  # P6
        [[0, 0], [1, 1]],  # P7
    ]
    strassen_z = [
        [1, 0, 0, -1, 1, 0, 1],   # z00: P1+P4-P5+P7
        [0, 0, 1, 0, 1, 0, 1],   # z01: P3+P5+P7
        [0, 1, 0, 1, 0, -1, 1],  # z10: P2+P4-P6+P7
        [1, -1, 1, 0, 1, 0, 0],  # z11: P1-P2+P3+P5
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A1*B11 -> C1
        (0, 2, 2, 0, 0, 0),  # A2*B21 -> C1
        (0, 0, 0, 2, 0, 2),  # A1*B12 -> C2
        (0, 2, 2, 2, 0, 2),  # A2*B22 -> C2
    ]
    
    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for t in range(7):
            col = b_idx * 7 + t
            for xi in range(2):
                for xj in range(2):
                    coeff = strassen_x[t][xi][xj]
                    if coeff != 0:
                        U[(ar + xi) * m + (ac + xj)][col] += coeff
            for yi in range(2):
                for yj in range(2):
                    coeff = strassen_y[t][yi][yj]
                    if coeff != 0:
                        V[(br + yi) * p + (bc + yj)][col] += coeff
            for zi in range(2):
                for zj in range(2):
                    coeff = strassen_z[zi][zj][t]
                    if coeff != 0:
                        W[(cc + zj) * n + (cr + zi)][col] += coeff
    
    return {"U": U, "V": V, "W": W}
