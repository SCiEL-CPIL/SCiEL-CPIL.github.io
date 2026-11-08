"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = 140.0  # safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # 6x6 matrix multiplication via 3x3 grid of 2x2 blocks.
    # Each 2x2 block multiply uses the rank-7 Strassen algorithm.
    # Total rank = 27 * 7 = 189.
    #
    # Factor matrices:
    #   U: (n*m, rank)  -- coefficients of A
    #   V: (m*p, rank)  -- coefficients of B
    #   W: (p*n, rank)  -- coefficients of C, indexed W[k*n + i, r]
    #     where i is the row of C and k is the column of C.
    #
    # Exactness: for each tensor term r, sum_r U[i,j,r]*V[j,k,r] = C[i,k]
    # The decomposition is valid when, for every output entry (i,k),
    # sum_r U[i,k_row..., r] * V[...k, r] * W[k*n+i, r] reconstructs C[i,k].

    block_size = 2
    blocks_per_dim = 3  # 6 / 2

    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen patterns for 2x2 matrices.
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (local row, local col) -> value  (for A block)
    # v_coeffs: dict (local row, local col) -> value  (for B block)
    # w_coeffs: dict (local row, local col) -> value  (for C block)
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
                    # Update U: A is n x m, index r*m + c, r = global row, c = global col
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val

                    # Update V: B is m x p, index r*p + c, r = global row, c = global col
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val

                    # Update W: C is n x p, index k*n + i, i = global row, k = global col
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val

                    col_idx += 1
                    # Deadline guard
                    if time.time() - t0 > _DEADLINE:
                        break
            else:
                continue
            break
        else:
            continue
        break
    else:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
