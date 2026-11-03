"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5

    # We'll build a list of terms. Each term is a tuple (u_vec, v_vec, w_vec)
    # where u_vec is a vector of length 8, v_vec is a vector of length 20, w_vec is a vector of length 10.
    # But we can build them incrementally using dicts.

    terms = []

    # Helper to add a schoolbook term for a 2x2*2x2 block.
    def add_schoolbook_2x2_2x2(a_rows, a_cols, b_rows, b_cols, c_cols):
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = b_cols
        # j1, j2 are also the row indices of B.
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    u = {i*m + j: 1}
                    v = {j*p + k: 1}
                    w = {k*n + i: 1}
                    terms.append((u, v, w))

    # Helper to add a schoolbook term for a 2x2*2x1 block.
    def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_col):
        i1, i2 = a_rows
        j1, j2 = a_cols
        k = b_col
        for i in [i1, i2]:
            for j in [j1, j2]:
                u = {i*m + j: 1}
                v = {j*p + k: 1}
                w = {k*n + i: 1}
                terms.append((u, v, w))

    # Add the 4 2x2*2x2 blocks.
    # Block 1: A cols 0,1 * B rows 0,1 cols 0,1
    add_schoolbook_2x2_2x2([0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    # Block 2: A cols 0,1 * B rows 0,1 cols 2,3
    add_schoolbook_2x2_2x2([0, 1], [0, 1], [0, 1], [2, 3], [2, 3])
    # Block 3: A cols 2,3 * B rows 2,3 cols 0,1
    add_schoolbook_2x2_2x2([0, 1], [2, 3], [2, 3], [0, 1], [0, 1])
    # Block 4: A cols 2,3 * B rows 2,3 cols 2,3
    add_schoolbook_2x2_2x2([0, 1], [2, 3], [2, 3], [2, 3], [2, 3])

    # Add the 2 2x2*2x1 blocks.
    # Block 5: A cols 0,1 * B rows 0,1 col 4
    add_schoolbook_2x2_2x1([0, 1], [0, 1], [0, 1], 4, 4)
    # Block 6: A cols 2,3 * B rows 2,3 col 4
    add_schoolbook_2x2_2x1([0, 1], [2, 3], [2, 3], 4, 4)

    # Now, we have a list of terms. Each term is (u, v, w) where u, v, w are dicts.
    # The total number of terms is 4*4 + 2*4 = 16 + 8 = 24.
    # So rank = 24.

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][col] += coeff
        for idx, coeff in v.items():
            V[idx][col] += coeff
        for idx, coeff in w.items():
            W[idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
