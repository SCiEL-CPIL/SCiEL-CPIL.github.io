import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 blocks
    # U, V, W are lists of (row, col, coeff)
    strassen_terms = [
        # M1
        ([((0, 0, 1), (1, 1, 1))], 
         [((0, 0, 1), (1, 1, 1))], 
         [((0, 0, 1), (1, 1, 1))]),
        # M2
        ([((1, 0, 1), (1, 1, 1))], 
         [((0, 0, 1))], 
         [((1, 0, 1), (1, 1, -1))]),
        # M3
        ([((0, 0, 1))], 
         [((0, 1, 1), (1, 1, -1))], 
         [((0, 1, 1), (1, 1, 1))]),
        # M4
        ([((1, 1, 1))], 
         [((1, 0, 1), (0, 0, -1))], 
         [((0, 0, 1), (1, 0, 1))]),
        # M5
        ([((0, 0, 1), (0, 1, 1))], 
         [((1, 1, 1))], 
         [((0, 0, -1), (0, 1, 1))]),
        # M6
        ([((1, 0, 1), (0, 0, -1))], 
         [((0, 0, 1), (0, 1, 1))], 
         [((1, 1, 1))]),
        # M7
        ([((0, 1, 1), (1, 1, -1))], 
         [((1, 0, 1), (1, 1, 1))], 
         [((0, 0, 1))])
    ]
    
    # Iterate over 3x3 grid of 2x2 blocks
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base_col = (i * 9 + j * 3 + k) * 7
                for t in range(7):
                    col_idx = base_col + t
                    term = strassen_terms[t]
                    
                    for lr, lc, coeff in term[0]: # U
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row][col_idx] += coeff
                    for lr, lc, coeff in term[1]: # V
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row][col_idx] += coeff
                    for lr, lc, coeff in term[2]: # W
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff

    return {"U": U, "V": V, "W": W}
