"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition.
SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N_DIM, M_DIM, P_DIM = 3, 3, 6


def _build_support(n, m, p):
    """Return (on, off) lists of (a,b,c) indices and precomputed per-row
    index groups (for each a/b/c which (b,c)/(a,c)/(a,b) pair with it)."""
    nm, mp, pn = n * m, m * p, p * n
    on = []
    off = []
    # precompute a,b,c index decoding
    ai = np.array([a // m for a in range(nm)])
    aj = np.array([a % m for a in range(nm)])
    bj = np.array([b // p for b in range(mp)])
    bk = np.array([b % p for b in range(mp)])
    ck = np.array([c // n for c in range(pn)])
    ci = np.array([c % n for c in range(pn)])

    # on-support
    on = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on.append((a, b, c))
    on = np.array(on, dtype=np.int64)

    # group: for each a, list of (b,c) in on-support
    a2bc = {a: [] for a in range(nm)}
    for (a, b, c) in on:
        a2bc[a].append((b, c))

    # for each b, list of (a,c) in on-support
    b2ac = {b: [] for b in range(mp)}
    for (a, b, c) in on:
        b2ac[b].append((a, c))

    # for each c, list of (a,b) in on-support
    c2ab = {c: [] for c in range(pn)}
    for (a, b, c) in on:
        c2ab[c].append((a, b))

    return on, a2bc, b2ac, c2ab, ai, aj, bj, bk, ck, ci


def _verify(U, V, W, n, m, p):
    """Exact (float) verification of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
        return False

    ai = [a // m for a in range(nm)]
    aj = [a % m for a in range(nm)]
    bj = [b // p for b in range(mp)]
    bk = [b % p for b in range(mp)]
    ck = [c // n for c in range(pn)]
    ci = [c % n for c in range(pn)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if not (aj[a] == bj[b] and bk[b] == ck[c] and ci[c] == ai[a]):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    return True


def _schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
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


def _grad_descent(U, V, W, on, a2bc, b2ac, c2ab, lr, n_iter, huber_c, deadline):
    """Smooth surrogate (Huber loss) projected gradient descent on U,V,W.
    Returns (U,V,W, loss)."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    on_a = on[:, 0]
    on_b = on[:, 1]
    on_c = on[:, 2]
    
    # Precompute index groups for faster gradient computation
    a_groups = {a: np.where(on_a == a)[0] for a in range(U.shape[0])}
    b_groups = {b: np.where(on_b == b)[0] for b in range(V.shape[0])}
    c_groups = {c: np.where(on_c == c)[0] for c in range(W.shape[0])}

    for it in range(n_iter):
        if time.time() > deadline:
            break

        # current tensor values on support
        # val[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Build matrices: for each on index t, M[t] = V[b_t] ⊙ W[c_t]
        VW = V[on_b] * W[on_c]  # (T, r)
        val = np.einsum('tr,ar->t', VW, U[on_a])  # (T,)
        res = val - 1.0  # want 0

        # Huber loss and grad w.r.t. U[a]
        # loss_t = 0.5*res^2 if |res|<c else c*|res| - 0.5 c^2
        # d loss_t / d res = res if |res|<c else c*sign(res)
        dres = np.where(np.abs(res) < huber_c, res, huber_c * np.sign(res))
        loss = 0.5 * np.sum(np.minimum(res ** 2, huber_c * np.abs(res)))

        # d loss / d U[a] = sum_t dres_t * V[b_t,r] W[c_t,r] for t with a_t=a
        gU = np.zeros_like(U)
        for a in range(U.shape[0]):
            idx = a_groups[a]
            if len(idx) > 0:
                gU[a] = (VW[idx] * dres[idx][:, None]).sum(axis=0)
        gV = np.zeros_like(V)
        for b in range(V.shape[0]):
            idx = b_groups[b]
            if len(idx) > 0:
                gV[b] = (U[on_a[idx]] * W[on_c[idx]] * dres[idx][:, None]).sum(axis=0)
        gW = np.zeros_like(W)
        for c in range(W.shape[0]):
            idx = c_groups[c]
            if len(idx) > 0:
                gW[c] = (U[on_a[idx]] * V[on_b[idx]] * dres[idx][:, None]).sum(axis=0)

        # gradient step with adaptive learning rate
        # Use smaller lr as we get closer to solution
        current_lr = lr * (0.995 ** it)
        scale = max(1e-8, np.sqrt(np.mean(U ** 2)))
        U -= current_lr * gU / scale
        V -= current_lr * gV / scale
        W -= current_lr * gW / scale

    # final loss
    VW = V[on_b] * W[on_c]
    val = np.einsum('tr,ar->t', VW, U[on_a])
    loss = 0.5 * np.sum((val - 1.0) ** 2)
    return U, V, W, loss


def _try_rank(n, m, p, r, on, a2bc, b2ac, c2ab, n_restarts, deadline, rng):
    """Try to find an exact decomposition of rank r. Returns dict or None."""
    nm, mp, pn = n * m, m * p, p * n
    on_a = on[:, 0]
    on_b = on[:, 1]
    on_c = on[:, 2]

    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        U = rng.standard_normal((nm, r)) * 0.5
        V = rng.standard_normal((mp, r)) * 0.5
        W = rng.standard_normal((pn, r)) * 0.5

        U, V, W, loss = _grad_descent(
            U, V, W, on, a2bc, b2ac, c2ab,
            lr=0.05, n_iter=400, huber_c=1.0, deadline=deadline - 2
        )

        if loss < 1e-6:
            # snap to half-integers
            U2 = np.round(U * 2) / 2
            V2 = np.round(V * 2) / 2
            W2 = np.round(W * 2) / 2
            VW = V2[on_b] * W2[on_c]
            val = np.einsum('tr,ar->t', VW, U2[on_a])
            if np.max(np.abs(val - 1.0)) < 1e-9:
                # full exact verify
                if _verify(U2, V2, W2, n, m, p):
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
            # also accept exact float solution
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None


def _build_offcheck(n, m, p):
    """Precompute all off-support (a,b,c) triples for exact verification speed."""
    nm, mp, pn = n * m, m * p, p * n
    off = []
    for a in range(nm):
        i_a, j_a = a // m, a % m
        for b in range(mp):
            j_b, k_b = b // p, b % p
            for c in range(pn):
                k_c, i_c = c // n, c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    off.append((a, b, c))
    return np.array(off, dtype=np.int64)


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6."""
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    n, m, p = N_DIM, M_DIM, P_DIM

    # guaranteed baseline
    base = _schoolbook(n, m, p)
    base_rank = n * m * p

    on, a2bc, b2ac, c2ab, ai, aj, bj, bk, ck, ci = _build_support(n, m, p)
    off = _build_offcheck(n, m, p)
    off_a, off_b, off_c = off[:, 0], off[:, 1], off[:, 2]

    rng = np.random.default_rng(12345)
    random.seed(12345)

    best = base
    best_rank = base_rank

    # Theoretical lower bound ~ n*p*m = 54 for this structure; but try lower.
    # Lower bound: max(nm, mp, pn) = max(9,18,18)=18, and also >= number of
    # distinct slices. We sweep upward from a small value.
    # Budget remaining after baseline:
    time_per_rank = 8  # seconds
    min_r = 18
    max_r = base_rank - 1  # we only care to beat schoolbook
    if time.time() > deadline - 10:
        return base

    # sweep ranks; for each, allow a few restarts within the time budget
    for r in range(min_r, base_rank):
        if time.time() > deadline - 6:
            break
        remaining = deadline - 6 - time.time()
        # how many restarts can we afford? each ~ time_per_rank/ n_restarts
        n_restarts = max(1, int(remaining / time_per_rank))
        n_restarts = min(n_restarts, 6)
        res = _try_rank(n, m, p, r, on, a2bc, b2ac, c2ab, n_restarts, deadline - 6, rng)
        if res is not None:
            best = res
            best_rank = r
            break  # found lowest-rank; stop (we sweep upward)

    # final safety: ensure best still verifies
    if time.time() > deadline - 3:
        return best
    try:
        if not _verify(best["U"], best["V"], best["W"], n, m, p):
            return base
    except Exception:
        return base
    return best
# EVOLVE-BLOCK-END
