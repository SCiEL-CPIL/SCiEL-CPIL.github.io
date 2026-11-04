import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 130.0


def _strassen_2x2():
    """Return alpha, beta, recon for 2x2 Strassen."""
    alpha = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    return alpha, beta, recon


def _build_77_rank():
    """Rank-77 block-Strassen decomposition for 4x4 * 4x6."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    alpha, beta, recon = _strassen_2x2()

    col = 0
    for s in range(7):
        # Part 1: inner Strassen for X_s(2x2) * Y_s[:,0:2](2x2)
        for si in range(7):
            # U: combine alpha[s] and alpha[si]
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    U[(ar * 2 + x_row) * m + (br * 2 + x_col), col] += ca * cr
            # V: combine beta[s] and beta[si]
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    V[(ac * 2 + y_row) * p + (bc * 3 + y_col), col] += cb * cc
            # W: combine recon[s] and recon[si]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    out_r = a * 2 + za
                    out_c = b * 3 + zb
                    W[out_c * n + out_r, col] += gamma * zgamma
            col += 1

        # Part 2: schoolbook for X_s(2x2) * Y_s[:,2](2x1)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    U[(ar * 2 + u) * m + (br * 2 + v), col] += cr
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    V[(ac * 2 + v) * p + (bc * 3 + 2), col] += cc
                for a, b, gamma in recon[s]:
                    out_r = a * 2 + u
                    out_c = b * 3 + 2
                    W[out_c * n + out_r, col] += gamma
                col += 1

    assert col == R, f"col={col} != R={R}"
    return U, V, W


def _build_schoolbook():
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W


def _verify(U, V, W, n, m, p, num_trials=10, seed=42):
    """Verify decomposition exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(num_trials):
        A = rng.integers(-9, 9, (n, m)).astype(float)
        B = rng.integers(-9, 9, (m, p)).astype(float)
        C_exp = A @ B
        C_rec = np.einsum('ikr,krr,irn->in', U, V, W)
        if not np.allclose(C_rec, C_exp, atol=1e-7):
            return False
    return True


def _to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row]
            for row in mat.tolist()]


def construct():
    n, m, p = DIMENSIONS

    try:
        U, V, W = _build_77_rank()
        if _verify(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass

    U, V, W = _build_schoolbook()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
