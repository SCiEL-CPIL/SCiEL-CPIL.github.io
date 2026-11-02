import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135


def _build_rank_32(n, m, p):
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


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using einsum."""
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


def _try_reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by removing terms."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_removal(indices_to_remove):
        mask = [i for i in range(rank) if i not in indices_to_remove]
        if len(mask) == 0:
            return None
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = len(mask)
            return (
                [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
            )
        return None

    # Try removing each single term
    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        result = check_removal([r])
        if result is not None:
            return result

    # Try removing pairs
    for r1 in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        for r2 in range(r1 + 1, rank):
            result = check_removal([r1, r2])
            if result is not None:
                return result

    # Try removing triples if we have time
    if time.time() - _START_TIME < _TIMEOUT - 15:
        for r1 in range(rank):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            for r2 in range(r1 + 1, rank):
                if time.time() - _START_TIME > _TIMEOUT - 5:
                    return None
                for r3 in range(r2 + 1, rank):
                    if time.time() - _START_TIME > _TIMEOUT - 5:
                        return None
                    result = check_removal([r1, r2, r3])
                    if result is not None:
                        return result

    return None


def construct():
    n, m, p = DIMENSIONS

    # Build rank-32 decomposition
    U, V, W = _build_rank_32(n, m, p)

    # Validate
    if _validate_fast(U, V, W, n, m, p):
        # Try to reduce rank
        reduced = _try_reduce_rank(U, V, W, n, m, p)
        if reduced is not None:
            U2, V2, W2 = reduced
            if _validate_fast(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}

        return {"U": U, "V": W, "W": W} if False else {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
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
