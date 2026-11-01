import time
import math
import numpy as np

N = 150
_DEADLINE = 142.0  # hard wall-clock budget (150s limit with margin)

# ------------------- quaternion (constraint-free) helpers -------------------

def _quat_norm(q):
    return np.sqrt(np.sum(q * q, axis=1, keepdims=True))

def _quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=1)

def _quat_conj(q):
    return np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=1)

def _quat_to_point(q):
    q = q / _quat_norm(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z),
                     2 * (x * y - w * z),
                     2 * (x * z + w * y)], axis=1)

def _point_to_quat(p):
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    w = np.sqrt(np.maximum(0.0, 1.0 + x))
    s = 2.0 / np.where(w > 1e-9, w, 1.0)
    return np.stack([w, s * y, s * z, 0.0], axis=1)

# ------------------- energy / gradient (quaternion space) -------------------

def _energy_grad_q(q):
    """Energy and gradient w.r.t. quaternion params (unconstrained)."""
    qn = _quat_norm(q)
    qn_safe = np.where(qn > 1e-12, qn, 1.0)
    qh = q / qn_safe[:, 0]
    pts = _quat_to_point(qh)
    n = len(pts)

    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    grad_pts = inv_d2 @ pts - pts * inv_d2.sum(axis=1, keepdims=True)
    energy = np.log(np.sqrt(d2)).sum()  # sum_{i<j} log(1/d)

    # dE/dq = (1/|q|) * Re[ conj(q) * (0, grad_pts) ]
    gp = np.zeros((n, 4))
    gp[:, 1] = grad_pts[:, 0]
    gp[:, 2] = grad_pts[:, 1]
    gp[:, 3] = grad_pts[:, 2]
    qc = _quat_conj(q)
    prod = _quat_mul(qc, gp)
    grad_q = prod[:, 0] / qn_safe[:, 0]
    return energy, grad_q

def _energy_fast_q(q):
    qh = q / _quat_norm(q)
    pts = _quat_to_point(qh)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    iu = np.triu_indices(len(pts), k=1)
    d = np.sqrt(np.maximum(d2[iu], 1e-15))
    return np.sum(np.log(1.0 / d))

# ------------------- L-BFGS with two-loop recursion -------------------

def _lbfgs(q0, time_limit, max_iter=400, m=16):
    start = time.time()
    q = q0.copy()
    energy, grad = _energy_grad_q(q)
    s_hist, y_hist = [], []

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.3:
            break

        # two-loop recursion
        qv = grad.copy()
        rho = []
        for k in range(len(s_hist) - 1, -1, -1):
            s_k, y_k = s_hist[k], y_hist[k]
            yy = np.sum(y_k * y_k)
            a = np.sum(s_k * qv) / (yy + 1e-30)
            rho.append(a)
            qv = qv - a * y_k
        r = qv
        d = r.copy()
        for k in range(len(rho)):
            s_k, y_k = s_hist[k], y_hist[k]
            yy = np.sum(y_k * y_k)
            b = np.sum(y_k * r) / (yy + 1e-30)
            d = d + (rho[k] - b) * s_k

        if np.sum(d * d) < 1e-28:
            break
        dd = np.sum(grad * d)
        if dd > 0:
            d = -d
            dd = -dd

        # backtracking line search
        alpha = 0.5
        best_e, best_q, best_g, best_a = energy, q, grad, 0.0
        for _ in range(40):
            nq = q + alpha * d
            ne, ng = _energy_grad_q(nq)
            if ne < best_e:
                best_e, best_q, best_g, best_a = ne, nq, ng, alpha
            if ne <= energy + 1e-4 * alpha * dd:
                break
            alpha *= 0.6
            if alpha < 1e-13:
                break
        if best_a == 0.0:
            break

        s_hist.append(best_q - q)
        y_hist.append(best_g - grad)
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
        q, energy, grad = best_q, best_e, best_g
        if np.linalg.norm(grad) < 1e-12:
            break
    return q, energy

# ------------------- Riemannian CG (tangent, robust) -------------------

def _exponential_map(pts, tv):
    nrm = np.linalg.norm(tv, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-15)
    return np.cos(nrm) * pts + (np.sin(nrm) / nrm) * tv

def _energy_grad_sphere(pts):
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    inv_d2 = 1.0 / np.maximum(d2, 1e-15)
    np.fill_diagonal(inv_d2, 0.0)
    grad = inv_d2 @ pts - pts * inv_d2.sum(axis=1, keepdims=True)
    grad -= np.sum(grad * pts, axis=1, keepdims=True) * pts
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(d2[iu], 1e-15))
    return np.sum(np.log(1.0 / d)), grad

def _rcg(pts, time_limit, max_iter=300):
    start = time.time()
    cur = pts.copy()
    energy, grad = _energy_grad_sphere(cur)
    d = -grad.copy()
    prev = grad.copy()
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.3:
            break
        alpha = 0.1
        be, bp, bg, ba = energy, cur, grad, 0.0
        for _ in range(40):
            np_ = _exponential_map(cur, alpha * d)
            ne, ng = _energy_grad_sphere(np_)
            if ne < be:
                be, bp, bg, ba = ne, np_, ng, alpha
            if ne > energy + 1e-4 * alpha * np.sum(grad * d):
                break
            alpha *= 0.7
            if alpha < 1e-13:
                break
        if ba == 0.0:
            break
        cur, energy, grad = bp, be, bg
        if np.linalg.norm(grad) < 1e-13:
            break
        num = np.sum(grad * (grad - prev))
        den = np.sum(prev * prev)
        beta = max(0.0, num / den) if den > 1e-30 and it > 0 else 0.0
        d = -grad + beta * d
        prev = grad.copy()
    return cur, energy

def _normalize(pts):
    return pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-15)

# ------------------- initializers -------------------

def _fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    p = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(p)

def _halton(n):
    def h(idx, base):
        f, r, i = 1.0, 0.0, idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    pts = []
    for i in range(1, n + 1):
        u1, u2 = h(i, 2), h(i, 3)
        th, ph = 2.0 * np.pi * u1, np.arccos(2.0 * u2 - 1.0)
        pts.append([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
    return _normalize(np.array(pts))

# ------------------- basin hopping -------------------

def _basin_hopping(pts, time_limit, max_restarts=6):
    start = time.time()
    cur = pts.copy()
    q = _point_to_quat(cur)
    q, energy = _lbfgs(q, time_limit=min(15, time_limit - 2))
    cur = _quat_to_point(q / _quat_norm(q))
    cur = _normalize(cur)
    energy = _energy_fast_q(q)
    best_pts, best_e = cur.copy(), energy

    for rh in range(max_restarts):
        tl = time_limit - (time.time() - start)
        if tl < 6:
            break
        rng = np.random.default_rng(rh + 7)
        scale = 0.25 * (0.8 ** rh)
        pert = rng.standard_normal((N, 3)) * scale
        pert -= np.sum(pert * cur, axis=1, keepdims=True) * cur
        np_ = _normalize(_exponential_map(cur, pert))
        q_ = _point_to_quat(np_)
        q_, ne = _lbfgs(q_, time_limit=min(12, tl - 3))
        np_ = _normalize(_quat_to_point(q_ / _quat_norm(q_)))
        ne = _energy_fast_q(q_)
        # Metropolis
        dE = ne - energy
        if dE < 0 or (np.random.rand() < math.exp(-dE / max(0.1, 0.5 * (0.85 ** rh)))):
            cur, energy = np_, ne
            if energy < best_e:
                best_e, best_pts = energy, cur.copy()
    return best_pts, best_e

# ------------------- main -------------------

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    inits = [_fibonacci(N), _halton(N)]
    for s in range(4):
        rng = np.random.default_rng(s)
        inits.append(_normalize(rng.standard_normal((N, 3))))
    for s in range(3):
        rng = np.random.default_rng(s + 50)
        inits.append(_normalize(_fibonacci(N) + rng.standard_normal((N, 3)) * 0.12))

    best_pts, best_e = None, float('inf')
    for idx, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
        budget = 16.0 if idx == 0 else 13.0
        try:
            op, oe = _basin_hopping(init, time_limit=budget)
            oe = _energy_fast_q(_point_to_quat(op))
            if oe < best_e:
                best_e, best_pts = oe, op.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci(N)
        best_e = _energy_fast_q(_point_to_quat(best_pts))

    # final high-precision polish (alternating L-BFGS + RCG)
    for polish in range(3):
        if time.time() > deadline - 6:
            break
        tl = deadline - time.time() - 3
        if tl < 4:
            break
        try:
            q = _point_to_quat(best_pts)
            q, _ = _lbfgs(q, time_limit=tl * 0.5, max_iter=300)
            p = _normalize(_quat_to_point(q / _quat_norm(q)))
            p, _ = _rcg(p, time_limit=tl * 0.45, max_iter=250)
            p = _normalize(p)
            e = _energy_fast_q(_point_to_quat(p))
            if e < best_e:
                best_e, best_pts = e, p
        except Exception:
            pass

    return _normalize(best_pts)
