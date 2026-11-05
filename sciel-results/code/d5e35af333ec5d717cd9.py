"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen (rank 7) at 2x2 block level x schoolbook (rank 27) for 3x3 = rank 189."""
    n, m, p = DIMENSIONS
    bs = 3  # block size
    
    # Strassen's 7 products for 2x2 block multiplication
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),       # M1 = (A11+A22)(B11+B22)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),                   # M2 = (A21+A22)B11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),                  # M3 = A11(B12-B22)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),                  # M4 = A22(B21-B11)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),                   # M5 = (A11+A12)B22
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),        # M6 = (A21-A11)(B11+B12)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),        # M7 = (A12-A22)(B21+B22)
    ]
    
    # Output block combinations
    output_combos = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],   # C11 = M1 + M4 - M5 + M7
        (0,1): [(2, 1), (4, 1)],                     # C12 = M3 + M5
        (1,0): [(1, 1), (3, 1)],                     # C21 = M2 + M4
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # C22 = M1 - M2 + M3 + M6
    }
    
    rank = 7 * 27  # = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s_idx, (left_combo, right_combo) in enumerate(strassen_products):
        for a_local in range(bs):
            for b_local in range(bs):
                for c_local in range(bs):
                    # U: left factor contributions
                    for (abr, abc), coeff_a in left_combo.items():
                        row_i = abr * bs + a_local
                        row_j = abc * bs + b_local
                        U[row_i * m + row_j][col] += coeff_a
                    
                    # V: right factor contributions
                    for (bbr, bbc), coeff_b in right_combo.items():
                        row_j = bbr * bs + b_local
                        row_k = bbc * bs + c_local
                        V[row_j * p + row_k][col] += coeff_b
                    
                    # W: output contributions
                    for (ocr, occ), terms in output_combos.items():
                        for (s, coeff_out) in terms:
                            if s == s_idx:
                                row_k = occ * bs + c_local
                                row_i = ocr * bs + a_local
                                W[row_k * n + row_i][col] += coeff_out
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
