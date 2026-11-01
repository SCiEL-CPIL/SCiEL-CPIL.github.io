"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)

    col = 0

    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    # M2 = (a10+a11)b00 -> contributes to c10, c11
    # M3 = a00(b01-b11) -> contributes to c01, c11
    # M4 = a11(b10-b00) -> contributes to c00, c10
    # M5 = (a00+a01)b11 -> contributes to c00, c01
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00

    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs) where each is dict: local_2x2_idx -> coeff
        # local_2x2_idx: r*2+c for r,c in {0,1}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7
    ]

    # Split the 4 columns of A (and rows of B) into two groups of 2:
    #   part 0: columns 0,1 of A; rows 0,1 of B
    #   part 1: columns 2,3 of A; rows 2,3 of B
    # Split the 8 columns of B (and rows of C) into four groups of 2:
    #   block_m 0: columns 0,1 of B; rows 0,1 of C
    #   block_m 1: columns 2,3 of B; rows 2,3 of C
    #   block_m 2: columns 4,5 of B; rows 4,5 of C
    #   block_m 3: columns 6,7 of B; rows 6,7 of C

    for part in range(2):
        for block_m in range(4):
            for A_coeffs, B_coeffs, C_coeffs in strassen_terms:
                # A: local (r,c) -> global (i, j) where i=r, j=part*2+c
                #    u = i*m + j = i*4 + part*2 + c
                for local_idx, coeff in A_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_j = part * 2 + c
                    global_u = global_i * m + global_j
                    U[global_u, col] += coeff

                # B: local (r,c) -> global (j, k) where j=part*2+r, k=block_m*2+c
                #    v = j*p + k = (part*2+r)*8 + block_m*2 + c
                for local_idx, coeff in B_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_j = part * 2 + r
                    global_k = block_m * 2 + c
                    global_v = global_j * p + global_k
                    V[global_v, col] += coeff

                # C: local (r,c) -> global (i, k) where i=r, k=block_m*2+c
                #    w = k*n + i = (block_m*2+c)*2 + r
                for local_idx, coeff in C_coeffs.items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_k = block_m * 2 + c
                    global_w = global_k * n + global_i
                    W[global_w, col] += coeff

                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
