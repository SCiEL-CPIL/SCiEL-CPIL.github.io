"""Matrix multiplication 2x2 * 2x10 with rank-minimal tensor decomposition.

Uses the known Strassen algorithm for 2x2 * 2x2, applied block-wise to
split the 2x10 matrix into five 2x2 blocks. Each block uses 7 scalar
multiplications, giving total rank 35.
"""
import numpy as np


def construct():
    n, m, p = 2, 2, 10
    # Split B (2x10) into five 2x2 blocks
    n_blocks = p // m  # 5 blocks of width 2
    rank = n_blocks * 7  # 35

    # U: shape (n*m, rank) = (4, 35)
    # V: shape (m*p, rank) = (20, 35)
    # W: shape (p*n, rank) = (20, 35)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen's 7 products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 - P4 + P5 + P7
    # C01 = P3 + P4
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Define the Strassen decomposition as a list of (U_coeffs, V_coeffs, W_coeffs)
    # For each product Pk, we have:
    #   U coefficients: which entries of A are used and with what sign
    #   V coefficients: which entries of B are used and with what sign
    #   W coefficients: which entries of C receive the product and with what sign

    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d  -> flat: 0,1,2,3
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h  -> flat: 0,1,2,3
    # C indices: (0,0), (0,1), (1,0), (1,1)          -> flat: 0,1,2,3

    # For each Strassen product, define:
    #   u: dict {a_flat_idx: coeff}
    #   v: dict {b_flat_idx: coeff}
    #   w: list of (c_flat_idx, coeff)

    strassen_products = [
        # P1 = (a+d)(e+h)
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': [(0, 1), (3, 1)]},
        # P2 = (c+d)e
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': [(2, 1)]},
        # P3 = a(f-h)
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': [(1, 1)]},
        # P4 = d(g-e)
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': [(0, -1), (1, 1), (2, 1)]},
        # P5 = (a+b)h
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': [(0, 1)]},
        # P6 = (c-a)(e+f)
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': [(3, 1)]},
        # P7 = (b-d)(g+h)
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': [(0, 1)]},
    ]

    for block in range(n_blocks):
        for prod_idx, prod in enumerate(strassen_products):
            col = block * 7 + prod_idx
            # A is the same for all blocks, so U coefficients are the same
            for a_idx, coeff in prod['u'].items():
                U[a_idx, col] += coeff

            # B is split into blocks; block b uses B columns 2*b..2*b+1
            for b_idx, coeff in prod['v'].items():
                # b_idx: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
                b_row = b_idx // 2
                b_col_in_block = b_idx % 2
                b_col = block * m + b_col_in_block
                V[b_row * p + b_col, col] += coeff

            # C output: block b contributes to C columns 2*b..2*b+1
            for c_idx, coeff in prod['w']:
                c_row = c_idx // 2
                c_col_in_block = c_idx % 2
                c_col = block * m + c_col_in_block
                # W index: k*n + i where k is column, i is row
                W[c_col * n + c_row, col] += coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
