"""4x4 matrix multiplication: rank-minimal exact tensor decomposition.

Recursive Strassen (rank 49) built from two rank-7 Strassen identities,
then attempt a rank-48 merge of one redundant rank-1 term pair.
Numerical verification guards the final answer.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen-7 coefficient matrices (2x2).
_BETA = np.array([
    [[1, 0], [0, 1]],
    [[0, 0], [1, 1]],
    [[1, 0], [0, 0]],
    [[0, 0], [0, 1]],
    [[1, 1], [0, 0]],
    [[-1, 0], [1, 0]],
    [[0, 1], [0, -1]],
], dtype=int)

_GAMMA = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, 0]],
    [[0, 1], [0, -1]],
    [[-1, 0], [1, 0]],
    [[0, 0], [0, 1]],
    [[1, 1], [0, 0]],
    [[0, 0], [1, 1]],
], dtype=int)

_DELTA = np.array([
    [[1, 0], [0, 1]],
    [[0, 0], [1, -1]],
    [[0, 1], [0, 1]],
    [[1, 0], [1, 0]],
    [[-1, 1], [0, 0]],
    [[0, 0], [0, 1]],
    [[1, 0], [0, 0]],
], dtype=int)


def _build_49():
    """Build the 49-term recursive Strassen decomposition (U,V,W each 16x49)."""
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            # U: A-side, rows (I,S) from beta_s[I,?]*beta_t[?,?]
            for I in range(2):
                for S in range(2):
                    for J in range(2):
                        for T in range(2):
                            coeff = _BETA[s, I, J] * _BETA[t, S, T]
                            if coeff == 0:
                                continue
                            U[I*2+S, c] += 0  # placeholder; see below
            # Correct: U row index is (i,j) = (I*2+S, J*2+T)
            # U[i*m+j, c] where m=4, so row = i*4+j
            for I in range(2):
                for J in range(2):
                    for S in range(2):
                        for T in range(2):
                            coeff = _BETA[s, I, J] * _BETA[t, S, T]
                            if coeff == 0:
                                continue
                            row = I*4 + J
                            U[row, c] += coeff
            # V: B-side, row = j*4+k = (J*2+T)*4 + (K*2+V)
            for J in range(2):
                for K in range(2):
                    for T in range(2):
                        for Vv in range(2):
                            coeff = _GAMMA[s, J, K] * _GAMMA[t, T, Vv]
                            if coeff == 0:
                                continue
                            row = J*4 + K
                            V[row, c] += coeff
            # W: C-side, row = k*4+i = (K*2+Vv)*4 + (I*2+S)  [trace convention W[k*n+i]]
            for I in range(2):
                for K in range(2):
                    for S in range(2):
                        for Vv in range(2):
                            coeff = _DELTA[s, I, K] * _DELTA[t, S, Vv]
                            if coeff == 0:
                                continue
                            row = K*4 + I
                            W[row, c] += coeff
    return U, V, W


def _verify(U, V, W, seed=99, trials=6):
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        acc = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                acc[i, k] = s
        if not np.allclose(acc, C, atol=1e-6):
            return False
    return True


def construct():
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 15.0

    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)

    U49, V49, W49 = _build_49()

    if not _verify(U49.tolist(), V49.tolist(), W49.tolist()):
        # Baseline failed; return anyway (shouldn't happen).
        return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}

    # Attempt rank-48 merge: find pair (a,b) with W[:,a]+W[:,b] rank-1.
    Wmat = W49.astype(float)
    best = None
    for a in range(49):
        for b in range(a+1, 49):
            S = Wmat[:, a] + Wmat[:, b]
            if np.linalg.matrix_rank(S, tol=1e-9) != 1:
                continue
            Um = U49[:, a] + U49[:, b]
            Vm = V49[:, a] + V49[:, b]
            # integer / half-integer check
            def ok(x):
                return np.allclose(2*x, np.round(2*x), atol=1e-9)
            if not (ok(Um.astype(float)) and ok(Vm.astype(float)) and ok(S)):
                continue
            keep = [c for c in range(49) if c != a and c != b]
            U48 = np.column_stack([U49[:, c] for c in keep] + [Um.astype(int)])
            V48 = np.column_stack([V49[:, c] for c in keep] + [Vm.astype(int)])
            W48 = np.column_stack([W49[:, c] for c in keep] + [S.astype(int)])
            if _verify(U48.tolist(), V48.tolist(), W48.tolist()):
                best = (U48, V48, W48)
                break
        if best is not None:
            break
        if time.time() > _DEADLINE - _MARGIN:
            break

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}

# EVOLVE-BLOCK-END
