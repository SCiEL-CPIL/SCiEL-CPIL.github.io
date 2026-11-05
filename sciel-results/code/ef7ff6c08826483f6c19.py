"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank 7 for 2x2 blocks of size 3x3
    # Total rank 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # 1: (0,0), 2: (0,1), 3: (1,0), 4: (1,1)
    # alpha[s][q] : coeff of A_q in M_s
    # beta[s][r]  : coeff of B_r in M_s
    # gamma[s][p] : coeff of M_s in C_p
    
    # s=1: M1 = (A11+A22)(B11+B22)
    # s=2: M2 = (A21+A22)B11
    # s=3: M3 = A11(B12-B22)
    # s=4: M4 = A22(B21-B11)
    # s=5: M5 = (A11+A12)B22
    # s=6: M6 = (A21-A11)(B11+B12)
    # s=7: M7 = (A12-A22)(B21+B22)
    
    # Coefficients
    # Using dict {block_id: coeff}
    alphas = [
        {1: 1, 4: 1}, # M1
        {3: 1, 4: 1}, # M2
        {1: 1},       # M3
        {4: 1},       # M4
        {1: 1, 2: 1}, # M5
        {3: 1, 1: -1},# M6
        {2: 1, 4: -1} # M7
    ]
    betas = [
        {1: 1, 4: 1}, # M1
        {1: 1},       # M2
        {2: 1, 4: -1},# M3
        {3: 1, 1: -1},# M4
        {4: 1},       # M5
        {1: 1, 2: 1}, # M6
        {3: 1, 4: 1}  # M7
    ]
    gammas = [
        {1: 1, 4: 1}, # C11, C22
        {3: 1, 4: -1},# C21, C22
        {2: 1, 4: 1}, # C12, C22
        {1: 1, 3: 1}, # C11, C21
        {1: -1, 2: 1},# C11, C12
        {4: 1},       # C22
        {1: 1}        # C11
    ]
    
    # Helper to get block row/col indices
    def get_block_indices(block_id):
        # 1: (0,0), 2: (0,1), 3: (1,0), 4: (1,1)
        b = block_id - 1
        return b // 2, b % 2
        
    col = 0
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # U
                    for q, coeff in alphas[s].items():
                        ib, jb = get_block_indices(q)
                        row_a = ib * 3 + a
                        col_a = jb * 3 + b
                        U[row_a * m + col_a][col] += coeff
                        
                    # V
                    for r, coeff in betas[s].items():
                        ib, jb = get_block_indices(r)
                        row_b = ib * 3 + b
                        col_b = jb * 3 + c
                        V[row_b * p + col_b][col] += coeff
                        
                    # W
                    for p_id, coeff in gammas[s].items():
                        ib, jb = get_block_indices(p_id)
                        row_c = ib * 3 + a
                        col_c = jb * 3 + c
                        # W index convention: k*n + i -> col_c * n + row_c
                        W[col_c * n + row_c][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
