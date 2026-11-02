"""Rank-49 Strassen decomposition for 4x4x4 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 decomposition: 7 products
    # Each product is (A_linear_form, B_linear_form)
    # A is 2x2, B is 2x2, result is 2x2
    _STRASSEN_2X2 = [
        # M1 = (A00+A11) * (B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (A10+A11) * B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # M3 = A00 * (B01 - B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        # M4 = A11 * (B10 - B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # M5 = (A00+A01) * B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M6 = (A10-A00) * (B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # M7 = (A01-A11) * (B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    # C combination for 2x2 Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    _STRASSEN_C_COMBINATION = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Top-level Strassen: 7 block products for 4x4
    # Each block product computes a 2x2 block of C
    # Block products at the 4x4 level:
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # P3 = A00 * (B01-B11)
    # P4 = A11 * (B10-B00)
    # P5 = (A00+A01) * B11
    # P6 = (A10-A00) * (B00+B01)
    # P7 = (A01-A11) * (B10+B11)
    #
    # Where A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # Similarly for B, and C blocks.

    # Top-level block combinations
    _TOP_BLOCK_PRODUCTS = [
        # P1 = (A00+A11) * (B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # P2 = (A10+A11) * B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # P3 = A00 * (B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        # P4 = A11 * (B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # P5 = (A00+A01) * B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # P6 = (A10-A00) * (B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # P7 = (A01-A11) * (B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    # Top-level C combination:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    _TOP_C_COMBINATION = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # For each top-level block product (i = 0..6), apply Strassen 2x2 (s = 0..6)
    for i in range(7):
        a_combo, b_combo = _TOP_BLOCK_PRODUCTS[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            # Build U coefficients
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # Build V coefficients
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

    # Build W coefficients
    for i in range(7):
        for s in range(7):
            r = i * 7 + s
            # Get the top-level weight for block product i
            for (cbr, cbc, c_list) in _TOP_C_COMBINATION:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue

                # Get the Strassen 2x2 weight for sub-product s
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue

                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
