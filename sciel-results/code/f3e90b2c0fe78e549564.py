"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val

    blocks = [
        (0, 0, 0), # I, J, K
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    for I, J, K in blocks:
        # M1
        add_term(
            [((I+0)*4 + (J+0), 1), ((I+1)*4 + (J+1), 1)],
            [((J+0)*4 + (K+0), 1), ((J+1)*4 + (K+1), 1)],
            [((K+0)*2 + (I+0), 1), ((K+1)*2 + (I+1), 1)],
            col
        )
        col += 1
        
        # M2
        add_term(
            [((I+1)*4 + (J+0), 1), ((I+1)*4 + (J+1), 1)],
            [((J+0)*4 + (K+0), 1)],
            [((K+0)*2 + (I+1), 1)],
            col
        )
        col += 1
        
        # M3
        add_term(
            [((I+0)*4 + (J+0), 1)],
            [((J+1)*4 + (K+0), 1), ((J+1)*4 + (K+1), -1)],
            [((K+1)*2 + (I+0), 1)],
            col
        )
        col += 1
        
        # M4
        add_term(
            [((I+1)*4 + (J+1), 1)],
            [((J+0)*4 + (K+1), 1), ((J+0)*4 + (K+0), -1)],
            [((K+0)*2 + (I+0), 1)],
            col
        )
        col += 1
        
        # M5
        add_term(
            [((I+0)*4 + (J+0), 1), ((I+0)*4 + (J+1), 1)],
            [((J+1)*4 + (K+1), 1)],
            [((K+0)*2 + (I+0), -1)],
            col
        )
        col += 1
        
        # M6
        add_term(
            [((I+1)*4 + (J+0), 1), ((I+0)*4 + (J+0), -1)],
            [((J+0)*4 + (K+0), 1), ((J+1)*4 + (K+0), 1)],
            [((K+1)*2 + (I+1), 1)],
            col
        )
        col += 1
        
        # M7
        add_term(
            [((I+0)*4 + (J+1), 1), ((I+1)*4 + (J+1), -1)],
            [((J+1)*4 + (K+1), 1), ((J+0)*4 + (K+1), 1)],
            [((K+0)*2 + (I+0), 1)],
            col
        )
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
