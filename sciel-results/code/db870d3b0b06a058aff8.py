import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0
HARD_DEADLINE = 145.0


def _strassen_2x2x2():
    """Return U, V, W for the 2x2x2 Strassen decomposition (rank 7).

    A order: a00,a01,a10,a11 ; B order: b00,b01,b10,b11 ; C order: c00,c01,c10,c11
    m1=(a00+a11)(b00+b11)  m2=(a10+a11)b00  m3=a00(b01-b11)  m4=a11(b10-b00)
    m5=(a00+a01)b11        m6=(a10-a00)(b00+b01)  m7=(a01-a11)(b10+b11)
    c00=m1+m4-m5+m7  c01=m3+m5  c10=m2+m4  c11=m1-m2+m3+m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W


def _build_base():
    """Build the verified rank-88 decomposition: 8 Strassen (i=0,1) + 16 schoolbook (i=2)."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    col = 0
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            for i in range(4):
                U[u_rows[i], col:col + 7] += np.array(U2[i])
                V[v_rows[i], col:col + 7] += np.array(V2[i])
                W[w_rows[i], col:col + 7] += np.array(W2[i])
            col += 7
    # schoolbook for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            for a, b in [(j0, k0), (j1, k0), (j0, k1), (j1, k1)]:
                U[2 * m + a, col] = 1
                V[a * p + b, col] = 1
                W[b * n + 2, col] = 1
                col += 1
    return U, V, W


def _tensor(U, V, W):
    """Return (n,m,p) tensor T[i,j,k]=sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    Ue = U.reshape(n, m, 1, r)
    Ve = V.reshape(1, m, p, r)
    We = W.reshape(n, 1, p, r)
    return np.einsum('imjr,jplr,nkpr->imk', Ue, Ve, We)


def _valid(U, V, W):
    return np.allclose(_tensor(U, V, W), 1.0, atol=1e-8)


def _to_list(mat):
    mr = np.round(mat * 2) / 2
    if np.allclose(mat, mr, atol=1e-9):
        return mr.tolist()
    return mat.tolist()


def _greedy_remove(U, V, W, deadline):
    """Remove individually-removable columns via per-column tensor deltas (fast)."""
    r = U.shape[1]
    keep = np.ones(r, dtype=bool)
    T = _tensor(U[:, keep], V[:, keep], W[:, keep])
    for _ in range(8):
        if time.time() > deadline:
            break
        changed = False
        for c in range(r):
            if time.time() > deadline:
                break
            if not keep[c]:
                continue
            keep[c] = False
            Tnew = T - np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            if np.allclose(Tnew, 1.0, atol=1e-8):
                T = Tnew
                changed = True
            else:
                keep[c] = True
        if not changed:
            break
    idx = np.where(keep)[0]
    return U[:, idx], V[:, idx], W[:, idx]


def _nullspace_reduce(U, V, W, deadline):
    """Find a vector t in the null-space of the column-contraction map and
    perturb columns (U+alpha*t, V+beta*t, W+gamma*t) to zero out a column."""
    r = U.shape[1]
    n, m, p = DIMENSIONS
    if r < 2:
        return U, V, W
    # linear map M: R^3 -> R^R, M(a,b,g) = a*V*W + b*U*W + g*U*V  (elementwise products)
    Uf, Vf, Wf = U.ravel(), V.ravel(), W.ravel()
    M = np.empty((r, 3))
    M[:, 0] = Vf * Wf
    M[:, 1] = Uf * Wf
    M[:, 2] = Uf * Vf
    # null-space of M (vectors t with M t = 0)
    try:
        _, s, Vh = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9 * (s[0] if s.size else 1.0)
        nz = np.where(s > tol)[0]
        if len(nz) < 3:
            return U, V, W
        # null-space basis = last (3 - len(nz)) right singular vectors
        ns = Vh[len(nz):].T  # (3, k)
        if ns.size == 0:
            return U, V, W
    except Exception:
        return U, V, W
    # try to zero out the column c with the largest |t| contribution
    best = None
    for c in range(r):
        if time.time() > deadline:
            break
        target = np.zeros(r)
        target[c] = 1.0
        # solve M x = target  (least squares); if residual ~0 we can zero column c
        try:
            x, res, rk, sv = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            continue
        if np.linalg.norm(M @ x - target) > 1e-6:
            continue
        # x is the (a,b,g) perturbation; apply
        U2 = U + x[0] * np.outer(U[:, c], np.ones(r))  # placeholder, corrected below
        # Correct: U' = U + a * (t_U) where we set t_U so that perturbation = a*V*W etc.
        # We need U'V'W' change = a*VW + b*UW + g*UV = - (col c contribution)
        # U' = U + a * u, V' = V + b * v, W' = W + g * w with u,v,w free.
        # Simplest: choose u=U[:,c], v=V[:,c], w=W[:,c] -> change = a*V_c W_c + b*U_c W_c + g*U_c V_c
        # which equals M @ (a,b,g) with M built from column c products. Recompute M for col c.
        Mc = np.empty((r, 3))
        Mc[:, 0] = V[:, c] * Wf
        Mc[:, 1] = U[:, c] * Wf
        Mc[:, 2] = U[:, c] * Vf
        try:
            xc, _, _, _ = np.linalg.lstsq(Mc, target, rcond=None)
        except Exception:
            continue
        if np.linalg.norm(Mc @ xc - target) > 1e-6:
            continue
        U2 = U + xc[0] * U[:, c][:, None]
        V2 = V + xc[1] * V[:, c][:, None]
        W2 = W + xc[2] * W[:, c][:, None]
        if _valid(U2, V2, W2):
            best = (U2, V2, W2, c)
            break
    if best is not None:
        U2, V2, W2, c = best
        keep = np.ones(r, dtype=bool)
        keep[c] = False
        idx = np.where(keep)[0]
        return U2[:, idx], V2[:, idx], W2[:, idx]
    return U, V, W


def construct():
    deadline = min(HARD_DEADLINE, START_TIME + TIME_LIMIT)
    U, V, W = _build_base()
    if not _valid(U, V, W):
        # fallback to pure schoolbook (always valid)
        n, m, p = DIMENSIONS
        rk = n * m * p
        U = np.zeros((n * m, rk)); V = np.zeros((m * p, rk)); W = np.zeros((p * n, rk))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1; V[j * p + k, c] = 1; W[k * n + i, c] = 1; c += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Phase 1: cheap greedy removal
    if time.time() < deadline:
        U, V, W = _greedy_remove(U, V, W, deadline)
    # Phase 2: null-space projection (structured)
    if time.time() < deadline:
        U, V, W = _nullspace_reduce(U, V, W, deadline)
    # Phase 3: another greedy sweep to mop up
    if time.time() < deadline:
        U, V, W = _greedy_remove(U, V, W, deadline)

    if not _valid(U, V, W):
        U, V, W = _build_base()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
