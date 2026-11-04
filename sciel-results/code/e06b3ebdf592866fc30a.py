"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    b = 3  # block size
    
    products = [
        {'X': [((0,0), 1), ((1,1), 1)], 'Y': [((0,0), 1), ((1,1), 1)], 'Z': [((0,0), 1), ((1,1), 1)]},
        {'X': [((1,0), 1), ((1,1), 1)], 'Y': [((0,0), 1)], 'Z': [((1,0), 1), ((1,1), -1)]},
        {'X': [((0,0), 1)], 'Y': [((0,1), 1), ((1,1), -1)], 'Z': [((0,1), 1), ((1,1), 1)]},
        {'X': [((1,1), 1)], 'Y': [((1,0), 1), ((0,0), -1)], 'Z': [((0,0), 1), ((1,0), 1)]},
        {'X': [((0,0), 1), ((0,1), 1)], 'Y': [((1,1), 1)], 'Z': [((0,0), -1), ((0,1), 1)]},
        {'X': [((1,0), 1), ((0,0), -1)], 'Y': [((0,0), 1), ((0,1), 1)], 'Z': [((1,1), 1)]},
        {'X': [((0,1), 1), ((1,1), -1)], 'Y': [((1,0), 1), ((1,1), 1)], 'Z': [((0,0), 1)]},
    ]
    
    # Compute rank
    rank = 0
    for prod in products:
        rank += b * b * b  # 27 per product
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        X_blocks = prod['X']
        Y_blocks = prod['Y']
        Z_blocks = prod['Z']
        
        for i in range(b):
            for j in range(b):
                for k in range(b):
                    for (xr, xc, xs) in X_blocks:
                        for (yr, yc, ys) in Y_blocks:
                            for (zr, zc, zs) in Z_blocks:
                                u_row = (xr * b + i) * m + (xc * b + j)
                                v_row = (yr * b + j) * p + (yc * b + k)
                                w_row = (zr * b + k) * n + (zc * b + i)
                                
                                val = xs * ys * zs
                                U[u_row][col] += val
                                V[v_row][col] += val
                                W[w_row][col] += val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
