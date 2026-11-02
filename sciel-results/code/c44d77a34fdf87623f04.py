"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using a block-Strassen approach.
    
    Partition each 6x6 matrix into 3x3 blocks of 2x2 submatrices.
    Apply Strassen's rank-7 decomposition to the 3x3 block multiplication.
    Total rank: 189.
    """
    n, m, p = DIMENSIONS
    nb = 3
    bs = 2
    
    rank = nb * nb * (nb * 7)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                for s in range(7):
                    if s == 0:
                        x_coeffs = {((0,0)): 1, ((1,1)): 1}
                        y_coeffs = {((0,0)): 1, ((1,1)): 1}
                        c_contrib = {((0,0)): 1, ((1,1)): 1}
                    elif s == 1:
                        x_coeffs = {((1,0)): 1, ((1,1)): 1}
                        y_coeffs = {((0,0)): 1}
                        c_contrib = {((1,0)): 1, ((1,1)): -1}
                    elif s == 2:
                        x_coeffs = {((0,0)): 1}
                        y_coeffs = {((0,1)): 1, ((1,1)): -1}
                        c_contrib = {((0,1)): 1, ((1,1)): 1}
                    elif s == 3:
                        x_coeffs = {((1,1)): 1}
                        y_coeffs = {((1,0)): 1, ((0,0)): -1}
                        c_contrib = {((0,0)): 1, ((1,0)): 1}
                    elif s == 4:
                        x_coeffs = {((0,0)): 1, ((0,1)): 1}
                        y_coeffs = {((1,1)): 1}
                        c_contrib = {((0,0)): -1, ((0,1)): 1}
                    elif s == 5:
                        x_coeffs = {((1,0)): 1, ((0,0)): -1}
                        y_coeffs = {((0,0)): 1, ((0,1)): 1}
                        c_contrib = {((1,1)): 1}
                    elif s == 6:
                        x_coeffs = {((0,1)): 1, ((1,1)): -1}
                        y_coeffs = {((1,0)): 1, ((1,1)): 1}
                        c_contrib = {((0,0)): 1}
                    
                    for (lr_a, lc_a), coeff_a in x_coeffs.items():
                        global_a_row = i_block * bs + lr_a
                        global_a_col = j_block * bs + lc_a
                        U[global_a_row * m + global_a_col][col] += coeff_a
                    
                    for (lr_b, lc_b), coeff_b in y_coeffs.items():
                        global_b_row = j_block * bs + lr_b
                        global_b_col = k_block * bs + lc_b
                        V[global_b_row * p + global_b_col][col] += coeff_b
                    
                    for (lr_c, lc_c), coeff_c in c_contrib.items():
                        global_c_row = i_block * bs + lr_c
                        global_c_col = k_block * bs + lc_c
                        w_row = global_c_col * n + global_c_row
                        W[w_row][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
