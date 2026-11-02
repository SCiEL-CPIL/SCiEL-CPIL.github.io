"""Fresh design: rank-reduction search for the 2x2 * 2x5 multiplication tensor.

Strategy: start from the known rank-20 (rank-17/18 family) exact decomposition,
then run a bounded alternating-least-squares search at rank 17 with exact
rational recovery. Return the best (lowest-rank) exact decomposition found,
falling back to the guaranteed rank-20 if the search fails or time runs out.
"""
import time
import fractions
import itertools
import numpy as np

DIMENSIONS = [2, 2, 5]

# Hard wall-clock deadline guard (evaluator kills at 150s; keep safety margin).
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _tens(T, i, j, k):
    # T[i*m+j, k*n+i] = 1  (official trace convention)
    n, m, p = DIMENSIONS
    return 1 if i == k else 0


def _build_tensor(n, m, p):
    T = np.zeros((n * m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-8):
    n, m, p = DIMENSIONS
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # reconstruct: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    r = U.shape[1]
    # Use einsum: U[a,c]*V[b,c]*W[c2,c] with a=i*m+j, b=j*p+k, c2=k*n+i
    # b = j*p + k, c2 = k*n + i. We need to match indices.
    # Build via loops for clarity (small dims).
    R = np.zeros((n * m, p * n))
    Uf = np.asarray(U, dtype=float)
    Vf = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c2 = k * n + i
                s = 0.0
                for c in range(r):
                    s += Uf[a, c] * Vf[b, c] * Wf[c2, c]
                R[a, c2] = s
    return np.max(np.abs(R - T)) <= tol


def _to_rational_matrix(M, max_den=1024):
    """Convert float matrix to a list-of-lists of Fraction (integer/half-integer)."""
    out = []
    for row in M:
        r = []
        for x in row:
            fr = fractions.Fraction(x).limit_denominator(max_den)
            r.append(fr)
        out.append(r)
    return out


def _check_rational_exact(Ur, Vr, Wr, T):
    """Exact rational check: sum_c Ur[a,c]*Vr[b,c]*Wr[c2,c] == T[a,c2]."""
    n, m, p = DIMENSIONS
    r = len(Ur[0])
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c2 = k * n + i
                s = fractions.Fraction(0, 1)
                for c in range(r):
                    s += Ur[a][c] * Vr[b][c] * Wr[c2][c]
                if s != fractions.Fraction(1 if i == k else 0, 1):
                    return False
    return True


def _als_step(U, V, W, T, n, m, p, r, iters=60, tol=1e-9):
    """Alternating least squares for rank-r decomposition of T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Uf = U.copy(); Vf = V.copy(); Wf = W.copy()
    best_err = float('inf')
    for it in range(iters):
        if _time_left() < 3.0:
            break
        # Solve U: for each a=i*m+j, U[a,:] solves  sum_c U[a,c]*Vf[b,c]*Wf[c2,c] = T[a,c2]
        # Build linear system per a: unknowns U[a,c]. For each (k) target c2=k*n+i,
        # coefficient for U[a,c] is Vf[j*p+k, c]*Wf[k*n+i, c].
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # System: for k in 0..p-1: sum_c U[a,c] * Vf[j*p+k,c]*Wf[k*n+i,c] = T[a, k*n+i]
                A = np.zeros((p, r))
                rhs = np.zeros(p)
                for k in range(p):
                    b = j * p + k
                    c2 = k * n + i
                    A[k, :] = Vf[b, :] * Wf[c2, :]
                    rhs[k] = T[a, c2]
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                Uf[a, :] = sol
        # Solve V: for each b=j*p+k, unknowns V[b,c].
        for j in range(m):
            for k in range(p):
                b = j * p + k
                # targets over (i): for i in 0..n-1: sum_c Uf[i*m+j,c]*V[b,c]*Wf[k*n+i,c] = T[i*m+j, k*n+i]
                A = np.zeros((n, r))
                rhs = np.zeros(n)
                for i in range(n):
                    a = i * m + j
                    c2 = k * n + i
                    A[i, :] = Uf[a, :] * Wf[c2, :]
                    rhs[i] = T[a, c2]
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                Vf[b, :] = sol
        # Solve W: for each c2=k*n+i, unknowns W[c2,c].
        for k in range(p):
            for i in range(n):
                c2 = k * n + i
                # targets over (j): for j in 0..m-1: sum_c Uf[i*m+j,c]*Vf[j*p+k,c]*W[c2,c] = T[i*m+j, k*n+i]
                A = np.zeros((m, r))
                rhs = np.zeros(m)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    A[j, :] = Uf[a, :] * Vf[b, :]
                    rhs[j] = T[a, c2]
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                Wf[c2, :] = sol
        # error
        err = 0.0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c2 = k * n + i
                    s = float(np.dot(Uf[a, :], Vf[b, :] * Wf[c2, :]))
                    err = max(err, abs(s - T[a, c2]))
        if err < best_err:
            best_err = err
            best = (Uf.copy(), Vf.copy(), Wf.copy())
        if err < tol:
            break
    return best, best_err


def _rank20_base():
    """Known exact rank-20 decomposition (n*m*p=20)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _random_rank17_start(n, m, p, seed):
    rng = np.random.default_rng(seed)
    r = 17
    U = rng.standard_normal((n * m, r))
    V = rng.standard_normal((m * p, r))
    W = rng.standard_normal((p * n, r))
    return U, V, W


def _perturb_rank18_to_17(U, V, W, n, m, p, seed):
    """Take a rank-18-ish structure and drop one column with small perturbation."""
    rng = np.random.default_rng(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # drop last column, add small noise to remaining
    U = U[:, :17].copy()
    V = V[:, :17].copy()
    W = W[:, :17].copy()
    U += 0.01 * rng.standard_normal(U.shape)
    V += 0.01 * rng.standard_normal(V.shape)
    W += 0.01 * rng.standard_normal(W.shape)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Guaranteed fallback: rank-20 schoolbook.
    U20, V20, W20 = _rank20_base()
    best = (U20, V20, W20)
    best_rank = 20

    # Try to find rank-17 exact decomposition via ALS + rational recovery.
    attempts = 0
    max_attempts = 40
    while _time_left() > 8.0 and attempts < max_attempts:
        attempts += 1
        if attempts <= 20:
            U, V, W = _random_rank17_start(n, m, p, seed=1000 + attempts)
        else:
            U, V, W = _perturb_rank18_to_17(U20, V20, W20, n, m, p, seed=2000 + attempts)
        res, err = _als_step(U, V, W, T, n, m, p, 17, iters=40, tol=1e-10)
        Uf, Vf, Wf = res
        if err < 1e-7:
            # Try exact rational recovery.
            Ur = _to_rational_matrix(Uf, max_den=256)
            Vr = _to_rational_matrix(Vf, max_den=256)
            Wr = _to_rational_matrix(Wf, max_den=256)
            if _check_rational_exact(Ur, Vr, Wr, T):
                return {"U": Ur, "V": Vr, "W": Wr}
            # Try a few more denoms
            for md in (512, 1024, 4096):
                Ur = _to_rational_matrix(Uf, max_den=md)
                Vr = _to_rational_matrix(Vf, max_den=md)
                Wr = _to_rational_matrix(Wf, max_den=md)
                if _check_rational_exact(Ur, Vr, Wr, T):
                    return {"U": Ur, "V": Vr, "W": Wr}

    # Return best found (rank-20 guaranteed).
    Ur = _to_rational_matrix(best[0], max_den=1024)
    Vr = _to_rational_matrix(best[1], max_den=1024)
    Wr = _to_rational_matrix(best[2], max_den=1024)
    if not _check_rational_exact(Ur, Vr, Wr, T):
        # Fallback to plain schoolbook integer form.
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
    return {"U": Ur, "V": Vr, "W": Wr}
