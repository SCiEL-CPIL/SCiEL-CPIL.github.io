import time

def construct():
    _t0 = time.time()
    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    # Use rank 63: 6 Strassen blocks (36 terms) + 6 1x2x2 blocks (24 terms) + 3 merged terms
    RANK = 63
    U = [[0] * RANK for _ in range(n * m)]  # (12, 63)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 63)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 63)

    col = 0

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook)."""
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen_merged(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 6 (merged Strassen).
        Uses 6 multiplications instead of 7 by merging M7 into M1 and M4.
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1' = (A00+A01)(B00+B11) -> C00, C11
        U[A00][col] = 1; U[A01][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00 -> C10, -C11
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11) -> C01, C11
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4' = A11(B10-B00) -> C00, C10
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11 -> -C00, C01
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01) -> C11
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

    # ---- Build the 63-term decomposition ----
    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks with merged Strassen) ---
        _add_strassen_merged(1, 0, 0, bc, 1, 2 * j)
        _add_strassen_merged(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    return {"U": U, "V": V, "W": W}
