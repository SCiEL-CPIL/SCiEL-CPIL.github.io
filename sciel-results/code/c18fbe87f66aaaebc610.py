import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _time_remaining():
    return _TIMEOUT - (time.time() - _START_TIME)


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove single or pair of terms that are linearly dependent."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # First try single-term removal
    for r in range(rank):
        if _time_remaining() < 5:
            break
        Ua_r = np.delete(Ua, r, axis=1)
        Va_r = np.delete(Va, r, axis=1)
        Wa_r = np.delete(Wa, r, axis=1)
        T_recon = np.einsum('ir,jr,kr->ijk', Ua_r, Va_r, Wa_r)
        if np.allclose(T, T_recon, atol=1e-10):
            # Build reduced matrices
            U_new = [[U[i][j] for j in range(rank) if j != r] for i in range(len(U))]
            V_new = [[V[i][j] for j in range(rank) if j != r] for i in range(len(V))]
            W_new = [[W[i][j] for j in range(rank) if j != r] for i in range(len(W))]
            return U_new, V_new, W_new

    # Try pair-term removal (two terms that together are redundant)
    for r1, r2 in combinations(range(rank), 2):
        if _time_remaining() < 3:
            break
        Ua_r = np.delete(Ua, [r1, r2], axis=1)
        Va_r = np.delete(Va, [r1, r2], axis=1)
        Wa_r = np.delete(Wa, [r1, r2], axis=1)
        T_recon = np.einsum('ir,jr,kr->ijk', Ua_r, Va_r, Wa_r)
        if np.allclose(T, T_recon, atol=1e-10):
            keep = [j for j in range(rank) if j != r1 and j != r2]
            U_new = [[U[i][j] for j in keep] for i in range(len(U))]
            V_new = [[V[i][j] for j in keep] for i in range(len(V))]
            W_new = [[W[i][j] for j in keep] for i in range(len(W))]
            return U_new, V_new, W_new

    return None


def construct():
    n, m, p = DIMENSIONS

    U, V, W = _build_rank_32(n, m, p)

    if not _validate(U, V, W, n, m, p):
        # Fallback to trivial rank
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

    # Try to reduce rank via term removal
    if _time_remaining() > 10:
        reduced = _try_remove_terms(U, V, W, n, m, p)
        if reduced is not None:
            U, V, W = reduced
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
