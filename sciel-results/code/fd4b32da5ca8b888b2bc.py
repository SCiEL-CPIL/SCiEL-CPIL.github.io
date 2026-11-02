"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13

    # Strategy: split the 13 columns into 6 full 2-column blocks + 1 single column.
    # For each 2-column block we use a 7-term Strassen-style scheme; the leftover
    # column uses 4 terms. Total rank = 6*7 + 4 = 46.
    #
    # We index the tensors as:
    #   U over (i, j) with index i*m + j   (i in {0,1}, j in {0,1})
    #   V over (j, k) with index j*p + k   (j in {0,1}, k in {0..12})
    #   W over (k, i) with index k*n + i   (k in {0..12}, i in {0,1})
    #
    # The matrix product is C[i,k] = sum_j A[i,j] * B[j,k].

    R = 46
    U = [[0] * R for _ in range(n * m)]   # 4 x 46
    V = [[0] * R for _ in range(m * p)]   # 26 x 46
    W = [[0] * R for _ in range(p * n)]   # 26 x 46

    def _emit(u_spec, v_spec, w_spec, pos):
        for idx, val in u_spec:
            U[idx][pos] += val
        for idx, val in v_spec:
            V[idx][pos] += val
        for idx, val in w_spec:
            W[idx][pos] += val

    # ---- Strassen block for columns (k, k+1) ----
    # A = [[a00, a01],[a10, a11]],  Bblock = [[b00, b01],[b10, b11]]
    # Cblock = [[c00, c01],[c10, c11]]
    # We use these 7 bilinear products (with the standard Strassen identities):
    #   m1 = (a00 + a11) * (b00 + b11)
    #   m2 = (a10 + a11) * b00
    #   m3 = a00 * (b01 - b11)
    #   m4 = a11 * (b10 - b00)
    #   m5 = (a00 + a01) * b11
    #   m6 = (a10 - a00) * (b00 + b01)
    #   m7 = (a01 - a11) * (b10 + b11)
    # with
    #   c00 = m1 - m4 + m6
    #   c01 = m3 + m5
    #   c10 = m2 + m4
    #   c11 = m1 - m3 + m7

    def _strassen_block(k_lo, k_hi, base):
        # V indices: row j, col k  ->  j*p + k
        # W indices: row k, col i  ->  k*n + i
        # U indices: row i, col j  ->  i*m + j
        b00, b01 = k_lo, k_hi
        b10, b11 = p + k_lo, p + k_hi
        c00, c01 = k_lo * n + 0, k_hi * n + 0
        c10, c11 = k_lo * n + 1, k_hi * n + 1

        # m1
        _emit([(0, 1), (3, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)], base + 0)
        # m2
        _emit([(2, 1), (3, 1)], [(b00, 1)], [(c10, 1)], base + 1)
        # m3
        _emit([(0, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, -1)], base + 2)
        # m4
        _emit([(3, 1)], [(b10, 1), (b00, -1)], [(c00, -1), (c10, 1)], base + 3)
        # m5
        _emit([(0, 1), (1, 1)], [(b11, 1)], [(c01, 1)], base + 4)
        # m6
        _emit([(2, 1), (0, -1)], [(b00, 1), (b01, 1)], [(c00, 1)], base + 5)
        # m7
        _emit([(1, 1), (3, -1)], [(b10, 1), (b11, 1)], [(c11, 1)], base + 6)

    # ---- single-column block (schoolbook, 4 terms) ----
    def _single_column(k, base):
        b00 = k
        b10 = p + k
        c00 = k * n + 0
        c10 = k * n + 1
        _emit([(0, 1)], [(b00, 1)], [(c00, 1)], base + 0)
        _emit([(1, 1)], [(b10, 1)], [(c00, 1)], base + 1)
        _emit([(2, 1)], [(b00, 1)], [(c10, 1)], base + 2)
        _emit([(3, 1)], [(b10, 1)], [(c10, 1)], base + 3)

    pos = 0
    for b in range(6):
        _strassen_block(2 * b, 2 * b + 1, pos)
        pos += 7
    _single_column(12, pos)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
