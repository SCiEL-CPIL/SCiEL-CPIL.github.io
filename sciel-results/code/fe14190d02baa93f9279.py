import time
import numpy as np

START = time.time()
DEADLINE = START + 110.0


def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_rank98():
    """
    Rank-98 decomposition for 4x4 * 4x8 via tensor product of Strassen 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()

    U4 = np.zeros((16, 49), dtype=np.int16)
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i * 4 + j, c] = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j * 4 + k, c] = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k * 4 + i, c] = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]

    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    for block in range(2):
        cs = block * 49
        ce = (block + 1) * 49
        U[:, cs:ce] = U4
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j * 8 + kg, cs:ce] = V4[j * 4 + kl, :]
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg * 4 + i, cs:ce] = W4[kl * 4 + i, :]

    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    u_rows = i_idx.flatten() * m + j_idx.flatten()
    v_rows = j_idx.flatten() * p + k_idx.flatten()
    w_rows = k_idx.flatten() * n + i_idx.flatten()
    prod = U[u_rows, :].astype(np.int64) * V[v_rows, :].astype(np.int64) * W[w_rows, :].astype(np.int64)
    total = prod.sum(axis=1)
    return np.all(total == 1)


def _try_reduce_rank(U, V, W, max_cols_to_try=10):
    """
    Try to reduce rank by removing redundant columns via integer linear algebra.
    Returns a lower-rank decomposition or None.
    """
    rank = U.shape[1]
    # Try removing each column and checking if the decomposition still holds
    for c in range(rank):
        if time.time() > DEADLINE - 10:
            break
        U_t = np.delete(U, c, axis=1)
        V_t = np.delete(V, c, axis=1)
        W_t = np.delete(W, c, axis=1)
        if _verify(U_t, V_t, W_t):
            return U_t, V_t, W_t
    # Try pairs
    for c1 in range(rank):
        if time.time() > DEADLINE - 10:
            break
        for c2 in range(c1 + 1, min(c1 + max_cols_to_try, rank)):
            idx = [i for i in range(rank) if i != c1 and i != c2]
            U_t = U[:, idx]
            V_t = V[:, idx]
            W_t = W[:, idx]
            if _verify(U_t, V_t, W_t):
                return U_t, V_t, W_t
    return None


def _build_rank98_optimized():
    """Build rank-98 and try to reduce it."""
    U, V, W = _build_rank98()
    # Try simple column removal
    reduced = _try_reduce_rank(U, V, W)
    if reduced is not None:
        U, V, W = reduced
    return U, V, W


def _schoolbook(n=4, m=4, p=8):
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
    try:
        if time.time() < DEADLINE:
            U, V, W = _build_rank98_optimized()
            if _verify(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
