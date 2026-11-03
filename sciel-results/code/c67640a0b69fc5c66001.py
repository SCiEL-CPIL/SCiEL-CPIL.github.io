"""Matrix multiplication 3,4,7 — rank-minimal exact tensor decomposition.

Key idea: the 3x4x7 multiplication tensor is a slice (k<7) of the 3x4x8
multiplication tensor, whose block structure (2 j-blocks x 4 k-blocks of
3x2x2 multiplications) gives an upper bound of 48. We search (via a small
integer/half-integer alphabet) for the minimal rank of the 3x4x8 tensor; any
valid 3x4x8 decomposition restricted to k<7 is a valid 3x4x7 decomposition
of the same rank.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 150.0
_SAFE = DEADLINE - 25.0  # wall-clock safety margin before the evaluator kills us


def _mul_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # build C[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _half_int_ok(x):
    return np.all(np.abs(x * 2.0 - np.round(x * 2.0)) < 1e-6)


def _round_half(x):
    return np.round(x * 2.0) / 2.0


def _solve_uv(W, T, n, m, p, r):
    # Solve sum_c (U[:,c] (x) V[:,c]) = T.reshape(n*m, m*p)  for given W.
    # Unknowns: U (n*m x r), V (m*p x r). Linear in the stacked Kronecker terms.
    Tmat = T.reshape(n * m, m * p)
    # Build linear system: for each c, term_c = vec(U[:,c]) (x) vec(V[:,c])
    # This is bilinear; handle by fixing one side. We fix V and solve U, then
    # fix U and solve V, iterate (alternating least squares).
    nm, mp = n * m, m * p
    # Initialize
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    rng = np.random.default_rng(0)
    for c in range(r):
        U[:, c] = rng.normal(size=nm)
        V[:, c] = rng.normal(size=mp)
    for _ in range(400):
        # Solve U given V: for each c, U[:,c] = argmin || Tmat - sum_c U[:,c] (x) V[:,c] ||
        # Build A_U (mp*nm x nm*r) rows indexed by (j,k),(i)
        A_U = np.zeros((mp * nm, nm * r))
        for c in range(r):
            for jj in range(mp):
                row = np.zeros(nm)
                row[jj] = 1.0
                # A_U[(jj, i), (i, c)] = V[jj, c]
                A_U[jj * nm:(jj + 1) * nm, c * nm:(c + 1) * nm] = V[jj, c] * np.eye(nm)
        b_U = Tmat.T.reshape(-1)  # (i, jj) ordering
        sol, *_ = np.linalg.lstsq(A_U, b_U, rcond=None)
        U = sol.reshape(nm, r)
        # Solve V given U: A_V (nm*mp x mp*r), rows (i, jj)
        A_V = np.zeros((nm * mp, mp * r))
        for c in range(r):
            for ii in range(nm):
                A_V[ii * mp:(ii + 1) * mp, c * mp:(c + 1) * mp] = U[ii, c] * np.eye(mp)
        b_V = Tmat.reshape(-1)  # (jj, i) -> flatten (jj*nm + i)? need consistent order
        # b_V index = (jj, i) -> jj*nm + i
        b_V = Tmat.T.reshape(-1)  # Tmat.T[i, jj] = Tmat[jj, i]; flatten (i*mp+? )
        # careful: we want b_V[jj*nm + i] = Tmat[jj, i]
        b_V = Tmat.reshape(-1)  # row-major: Tmat[jj, i] index jj*nm + i  -> matches
        sol, *_ = np.linalg.lstsq(A_V, b_V, rcond=None)
        V = sol.reshape(mp, r)
    return U, V


def _construct_rank(W, n, m, p, r):
    U, V = _solve_uv(W, _mul_tensor(n, m, p), n, m, p, r)
    U = _round_half(U)
    V = _round_half(V)
    if not (_half_int_ok(U) and _half_int_ok(V) and _half_int_ok(W)):
        return None
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None


def _rand_W(n, m, p, r, rng):
    alphabet = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    W = np.zeros((p * n, r))
    # each output (k,i) must be reachable; give each column a sparse pattern
    for c in range(r):
        # pick a subset of (k,i) to be nonzero
        idx = rng.choice(p * n, size=rng.integers(1, 5), replace=False)
        for ii in idx:
            W[ii, c] = alphabet[rng.integers(0, len(alphabet))]
    return W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    # Best known: schoolbook for 3x4x7
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
    best = (rank, U, V, W)
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    # Search over 3x4x8 tensor (slice gives 3x4x7), ranks below 48.
    # Block upper bound for 3x4x8 = 4 * (3*2*2) = 48.
    T8 = _mul_tensor(n, m, 8)
    for r in range(48, max(best[0] - 1, 40) - 1, -1):
        if r >= best[0]:
            continue
        if time.time() > _SAFE:
            break
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            attempts += 1
            if time.time() > _SAFE:
                break
            W = _rand_W(n, m, 8, r, np_rng)
            res = _construct_rank(W, n, m, 8, r)
            if res is not None:
                U8, V8, W8 = res
                # Restrict to k<7: keep only rows of V with j*p8+k where k<7,
                # and rows of W with k<7.
                U = U8
                V = V8[:m * 7]  # rows (j*8+k) for k<7, j<4 -> first 4*7=28 rows
                W = W8[:7 * n]  # rows (k*3+i) for k<7 -> first 7*3=21 rows
                if _verify(U, V, W, n, m, p):
                    # convert to list of lists of int/half-int
                    def to_list(M):
                        return [[(int(x) if x == int(x) else x) for x in row] for row in M.tolist()]
                    best = (r, to_list(U), to_list(V), to_list(W))
                    break
        if best[0] == r:
            break
    return {"U": best[1], "V": best[2], "W": best[3]}
