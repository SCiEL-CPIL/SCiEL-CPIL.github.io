"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strategy: Use Strassen decomposition on 2x2 blocks of size 3.
    # 6x6 matrix is 3x3 blocks of 2x2.
    # 27 block multiplications, each of rank 7.
    # Total rank = 27 * 7 = 189.
    #
    # Factor matrices:
    #   U: (n*m, rank)  -- coefficients of A, indexed U[i*m + k, r]
    #   V: (m*p, rank)  -- coefficients of B, indexed V[k*p + j, r]
    #   W: (p*n, rank)  -- coefficients of C, indexed W[k*n + i, r]
    #     where i is the row of C and k is the column of C.

    block_size = 2
    blocks_per_dim = 3  # 6 / 2

    rank = (blocks_per_dim ** 3) * 7

    # Pre-allocate as numpy arrays for faster construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen patterns for 2x2 matrices.
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (local row, local col) -> value  (for A block)
    # v_coeffs: dict (local row, local col) -> value  (for B block)
    # w_coeffs: dict (local row, local col) -> value  (for C block)
    #
    # Strassen's 7 multiplications:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6

    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M1 = (A10+A11)B00
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},
         {(1, 0): 1, (1, 1): -1}),
        # M2 = A00(B01-B11)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1},
         {(0, 1): 1, (1, 1): 1}),
        # M3 = A11(B10-B00)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M4 = (A00+A01)B11
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},
         {(0, 0): -1, (0, 1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]

    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U: A block at (I, J)
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c, col_idx] += val

                    # Update V: B block at (J, K)
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c, col_idx] += val

                    # Update W: C block at (I, K)
                    # W index is k*n + i where k is col of C, i is row of C
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i, col_idx] += val

                    col_idx += 1

    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
