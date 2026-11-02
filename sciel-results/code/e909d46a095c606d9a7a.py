import time
import numpy as np

DIMENSIONS = [2, 2, 3]
N, M, P = 2, 2, 3


def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]


def _residual(U, V, W, n, m, p):
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)


def _softmax_sq(res, scale=40.0):
    """Smooth epigraph surrogate: log-sum-exp of squared residuals (scaled)."""
    x = (scale * res * res)
    mx = np.max(x)
    return (mx + np.log(np.sum(np.exp(x - mx)))) / scale


def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U


def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V


def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W


def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W


def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _multitensor(n, m, p):
    """Flatten the multiplication tensor (size n*m*p) with entries =1 at (i,j,k)."""
    T = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m*p + j*p + k] = 1.0
    return T


def _svd_init(n, m, p, r, seed):
    """Initialize rank-r factors from the dominant singular modes of the tensor."""
    rng = np.random.RandomState(seed)
    T = _multitensor(n, m, p)
    shape = (n*m, m*p, p*n)
    # Build a 3-way tensor with consistent index ordering.
    T3 = np.zeros(shape)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T3[i*m+j, j*p+k, k*n+i] = 1.0
    # Unfold along each mode and take top-r singular vectors.
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    for mode in range(3):
        mat = np.zeros((shape[mode], int(np.prod(shape) // shape[mode])))
        mat = np.reshape(T3, (shape[mode], -1))
        try:
            _, _, Vt = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            Vt = rng.uniform(-1, 1, (mat.shape[1], mat.shape[1]))
        top = min(r, Vt.shape[0])
        if mode == 0:
            U[:, :top] = Vt[:top].T
        elif mode == 1:
            V[:, :top] = Vt[:top].T
        else:
            W[:, :top] = Vt[:top].T
    # Fill any remaining columns with small random noise.
    if r > top:
        U[:, top:] = rng.uniform(-0.1, 0.1, (n*m, r-top))
        V[:, top:] = rng.uniform(-0.1, 0.1, (m*p, r-top))
        W[:, top:] = rng.uniform(-0.1, 0.1, (p*n, r-top))
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank(r, deadline, max_restarts=200, use_svd=True):
    n, m, p = 2, 2, 3
    best_res = float('inf')
    best_sol = None
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        seed = 1000 + r * 100 + restart
        if use_svd:
            U0, V0, W0 = _svd_init(n, m, p, r, seed)
        else:
            rng = np.random.RandomState(seed)
            U0 = rng.uniform(-1, 1, (n*m, r)).tolist()
            V0 = rng.uniform(-1, 1, (m*p, r)).tolist()
            W0 = rng.uniform(-1, 1, (p*n, r)).tolist()
        U, V, W = _als_optimize(U0, V0, W0, n, m, p, r, iters=60, deadline=deadline)
        res = _residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(res)
        if res_norm < best_res:
            best_res = res_norm
            best_sol = (U, V, W)
        if res_norm < 1e-6:
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
    return None


def _polish(U, V, W, n, m, p, r, deadline, steps=200):
    """Gradient-free coordinate polish with decreasing step + final half-integer snap."""
    best = (U, V, W)
    best_obj = _softmax_sq(_residual(U, V, W, n, m, p))
    for it in range(steps):
        if time.time() > deadline:
            break
        step = 0.5 / (1.0 + it)
        # coordinate ascent on a random block
        U2, V2, W2 = U, V, W
        for c in range(r):
            for i in range(n*m):
                for delta in (step, -step):
                    U2 = [row[:] for row in U]
                    U2[i][c] = U[i][c] + delta
                    obj = _softmax_sq(_residual(U2, V, W, n, m, p))
                    if obj < best_obj:
                        best_obj = obj
                        U = U2
                        best = (U, V, W)
        for c in range(r):
            for j in range(m*p):
                for delta in (step, -step):
                    V2 = [row[:] for row in V]
                    V2[j][c] = V[j][c] + delta
                    obj = _softmax_sq(_residual(U, V2, W, n, m, p))
                    if obj < best_obj:
                        best_obj = obj
                        V = V2
                        best = (U, V, W)
        for c in range(r):
            for k in range(p*n):
                for delta in (step, -step):
                    W2 = [row[:] for row in W]
                    W2[k][c] = W[k][c] + delta
                    obj = _softmax_sq(_residual(U, V, W2, n, m, p))
                    if obj < best_obj:
                        best_obj = obj
                        W = W2
                        best = (U, V, W)
        # exact local solve (ALS) to reduce residual
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
        obj = _softmax_sq(_residual(U, V, W, n, m, p))
        if obj < best_obj:
            best_obj = obj
            best = (U, V, W)
    return best


def _try_rank10_structured(deadline):
    """Exploit the 3rd-column structure: 4 fixed pure terms + a rank-6 2x2x2 free part.
    Parameterize the free part as a 2x2x2 tensor (rank 6, Strassen-like) and search
    for a decomposition that also satisfies the 2x3 extension constraints."""
    n, m, p = 2, 2, 3
    # Fixed terms for column k=2 (each involves a distinct W row):
    #   (i=0,j=0): U00 * V02 * W20
    #   (i=0,j=1): U01 * V12 * W20
    #   (i=1,j=0): U10 * V02 * W21
    #   (i=1,j=1): U11 * V12 * W21
    # We let these be parameterized too, and add 6 free terms for the 2x2 block.
    rng = np.random.RandomState(7)
    best_res = float('inf')
    best_sol = None
    for restart in range(80):
        if time.time() > deadline:
            break
        # 10 terms total. Build a generic rank-10 tensor but seed the fixed structure.
        r = 10
        U = rng.uniform(-1, 1, (n*m, r))
        V = rng.uniform(-1, 1, (m*p, r))
        W = rng.uniform(-1, 1, (p*n, r))
        # enforce fixed structure in last 4 terms
        U[0, 6] = 1.0; V[2, 6] = 1.0; W[4, 6] = 1.0   # (i0,j0,k2) -> W row k*n+i = 2*2+0=4
        U[1, 7] = 1.0; V[3, 7] = 1.0; W[4, 7] = 1.0   # (i0,j1,k2)
        U[2, 8] = 1.0; V[2, 8] = 1.0; W[5, 8] = 1.0   # (i1,j0,k2) -> W row 5
        U[3, 9] = 1.0; V[3, 9] = 1.0; W[5, 9] = 1.0   # (i1,j1,k2)
        # zero out the fixed W rows elsewhere to keep them pure
        for c in range(6):
            W[4, c] = 0.0
            W[5, c] = 0.0
        U, V, W = _als_optimize(U.tolist(), V.tolist(), W.tolist(), n, m, p, r,
                                 iters=40, deadline=deadline)
        res = _residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(res)
        if res_norm < best_res:
            best_res = res_norm
            best_sol = (U, V, W)
        if res_norm < 1e-6:
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    # Guaranteed fallback: rank 11
    best = _rank11_solution()

    # Try rank 10 with structured + SVD + random initializations
    if time.time() < deadline:
        sol = _try_rank10_structured(deadline)
        if sol:
            return {"U": sol[0], "V": sol[1], "W": sol[2]}

    if time.time() < deadline:
        sol = _try_rank(10, deadline, max_restarts=120, use_svd=True)
        if sol:
            return {"U": sol[0], "V": sol[1], "W": sol[2]}

    # Try rank 9
    if time.time() < deadline:
        sol = _try_rank(9, deadline, max_restarts=120, use_svd=True)
        if sol:
            return {"U": sol[0], "V": sol[1], "W": sol[2]}

    # Try rank 8
    if time.time() < deadline:
        sol = _try_rank(8, deadline, max_restarts=60, use_svd=True)
        if sol:
            return {"U": sol[0], "V": sol[1], "W": sol[2]}

    return best
