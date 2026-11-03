import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Index conventions: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    prods = [
        # M1
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        # M2
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        # M3
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        # M4
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        # M5
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        # M6
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        # M7
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Block structure:
    # A is 4x4  -> 2x2 blocks of 2x2 (I in {0,1}, J in {0,1})
    # B is 4x6  -> 2x3 blocks of 2x2 (J in {0,1}, K in {0,1,2})
    # C is 4x6  -> 2x3 blocks of 2x2 (I in {0,1}, K in {0,1,2})
    #
    # C_IK = A_I0 * B_0K + A_I1 * B_1K   (sum over J=0,1)
    #
    # Apply Strassen over the J-dimension for each fixed (I, K).
    # Strassen factors are the same for every (I,K) because the
    # J-block structure is identical; only the K-index of B and the
    # (I,K) index of C change. So for each (I,K) we need 7 rank terms.
    # Total rank = 2 (I) * 3 (K) * 7 (Strassen) = 42.

    U_s, V_s, W_s = _get_strassen_2x2_factors()

    rank = 2 * 3 * 7  # 42
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42

    col = 0
    for I in range(2):
        for K in range(3):
            for s in range(7):
                # U: A local (ii, jj) -> global i = 2*I+ii, j = 2*jj
                for ii in range(2):
                    for jj in range(2):
                        u_val = U_s[ii * 2 + jj][s]
                        if u_val != 0.0:
                            gi = 2 * I + ii
                            gj = 2 * jj
                            U[gi * m + gj][col] += u_val
                # V: B local (jj, kk) -> global j = 2*jj, k = 2*K+kk
                for jj in range(2):
                    for kk in range(2):
                        v_val = V_s[jj * 2 + kk][s]
                        if v_val != 0.0:
                            gj = 2 * jj
                            gk = 2 * K + kk
                            V[gj * p + gk][col] += v_val
                # W: C local (ii, kk) -> global i = 2*I+ii, k = 2*K+kk
                for ii in range(2):
                    for kk in range(2):
                        w_val = W_s[kk * 2 + ii][s]
                        if w_val != 0.0:
                            gi = 2 * I + ii
                            gk = 2 * K + kk
                            W[gk * n + gi][col] += w_val
                col += 1

    # Verify with a safety margin before the deadline.
    if time.time() < deadline:
        err = _verify(U, V, W, n, m, p, rank)
        if err > 1e-6:
            return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
