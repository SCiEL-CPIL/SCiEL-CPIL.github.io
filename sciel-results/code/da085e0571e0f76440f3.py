import time
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = 145.0
    n, m, p = 3, 4, 6
    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    col = 0

    def _idx_u(i, k): return i * m + k
    def _idx_v(k, j): return k * p + j
    def _idx_w(j, i): return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
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

        # M1
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1; col += 1
        # M2
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1; col += 1
        # M3
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1; col += 1
        # M4
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1; col += 1
        # M5
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1; col += 1
        # M6
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1; col += 1
        # M7
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; col += 1

    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK}, got {col}"
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Should equal 1 for all valid (i,k,j)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = sum(U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r] for r in range(RANK))
                assert abs(val - 1.0) < 1e-9, f"Tensor mismatch at ({i},{k},{j}): {val}"
    
    return {"U": U, "V": V, "W": W}
