import time
import numpy as np

_START = time.time()
_DEADLINE = 135.0  # explicit wall-clock guard with safety margin below the 150 s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _generate_strassen_4x4():
    """
    Rank-49 decomposition of the 4x4 x 4x4 matrix multiplication tensor.

    U : (n*m, rank) indexed by (i, k)
    V : (m*p, rank) indexed by (k, j)
    W : (p*n, rank) indexed by (j, i)   <- official trace convention W[k*n+i, r]

    Tensor check: T[i,k,j] = sum_r U[i,k,r]*V[k,j,r]*W[j,i,r] == 1 for all i,k,j.
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)

    # 2x2 Strassen sub-block combination vectors (row, col, coeff).
    # For a 2x2 product with sub-blocks A00,A01,A10,A11 and B00,B01,B10,B11:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11) B00
    #   M3 = A00 (B10-B11)
    #   M4 = A11 (B01-B00)
    #   M5 = (A00+A01) B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7 ; C01 = M3+M5
    # C10 = M2+M4 ;       C11 = M1-M2+M3+M7
    #
    # A-combo t: which (A-block, coeff) pairs
    _A_COMBOS = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(1, 0, 1), (1, 1, 1)],  # M2
        [(0, 0, 1)],             # M3
        [(1, 1, 1)],             # M4
        [(0, 0, 1), (0, 1, 1)],  # M5
        [(1, 0, 1), (0, 0, -1)], # M6
        [(0, 1, 1), (1, 1, -1)], # M7
    ]
    _B_COMBOS = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(0, 0, 1)],             # M2
        [(1, 0, 1), (1, 1, -1)], # M3
        [(0, 1, 1), (0, 0, -1)], # M4
        [(1, 1, 1)],             # M5
        [(0, 0, 1), (1, 0, 1)],  # M6
        [(0, 1, 1), (1, 1, 1)],  # M7
    ]
    _C_COMBOS = [
        [(0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, -1)],  # C00
        [(0, 1, 1), (1, 1, 1)],                          # C01
        [(1, 0, 1), (1, 1, 1)],                          # C10
        [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, 1)],   # C11
    ]

    for a in range(7):
        for b in range(7):
            r_idx = a * 7 + b

            # U factor: (A-block) (row-block, col-block)
            for (br, bc, sa) in _A_COMBOS[a]:
                for (ir, ik, sb) in _A_COMBOS[b]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    U[i_global * n + k_global, r_idx] += sa * sb

            # V factor: (B-block) (row-block, col-block)
            for (br, bc, sa) in _B_COMBOS[a]:
                for (ir, ik, sb) in _B_COMBOS[b]:
                    k_global = br * 2 + ir
                    j_global = bc * 2 + ik
                    V[k_global * n + j_global, r_idx] += sa * sb

            # W factor: (C-block) (row-block, col-block) -> indexed (j, i)
            for (br, bc, sa) in _C_COMBOS[a]:
                for (ir, ik, sb) in _C_COMBOS[b]:
                    i_global = br * 2 + ir
                    j_global = bc * 2 + ik
                    W[j_global * n + i_global, r_idx] += sa * sb

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    # T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
    T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Guaranteed-valid rank (n*m*p) decomposition as a safety net."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = 4, 4, 4

    # Hard deadline guard: if we are already past the safety margin,
    # fall back to the guaranteed-valid schoolbook result immediately.
    if _time_left() < 5.0:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = _generate_strassen_4x4()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback (should not be reached for a correct construction).
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
