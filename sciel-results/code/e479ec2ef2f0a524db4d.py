"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_t0 = time.time()
_DEADLINE = time.time() + 138.0  # safety margin before 150s kill


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return pts / n


def _covering_radius_deg(pts):
    """Covering radius via convex hull: largest circumradius of facets."""
    pts = _normalize(np.asarray(pts, dtype=np.float64))
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    simplices = hull.simplices
    v0 = pts[simplices[:, 0]]
    v1 = pts[simplices[:, 1]]
    v2 = pts[simplices[:, 2]]
    c = v0 + v1 + v2
    cn = np.linalg.norm(c, axis=1)
    valid = cn > 1e-10
    if not np.any(valid):
        return 90.0
    cc = c[valid] / cn[valid, None]
    a0 = np.clip(np.einsum('ij,ij->i', cc, v0[valid]), -1.0, 1.0)
    a1 = np.clip(np.einsum('ij,ij->i', cc, v1[valid]), -1.0, 1.0)
    a2 = np.clip(np.einsum('ij,ij->i', cc, v2[valid]), -1.0, 1.0)
    cr = np.max(np.arccos(np.stack([a0, a1, a2], axis=1)), axis=1)
    return float(np.degrees(np.max(cr)))


def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _random_init(seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)


def _smooth_obj_and_grad(pts, lam=0.6):
    """Objective: max pairwise angular distance + variance of squared distances.

    Smooth surrogate that pushes points apart uniformly (enforces uniformity via
    spectral/Gram properties) without a convex hull. Returns (value, gradient).
    """
    pts = _normalize(np.asarray(pts, dtype=np.float64))
    G = np.clip(pts @ pts.T, -1.0, 1.0)
    ang = np.arccos(G)
    iu = np.triu_indices(N, 1)
    d = ang[iu]
    mean = d.mean()
    var = d.var()
    # max term via log-sum-exp
    m = d.max()
    lse = m + np.log(np.sum(np.exp(d - m)))
    val = lse + lam * var
    # gradient
    sd = 1.0 / N * np.ones((N, N))
    # d(mean)/d ang = 1 / num_pairs
    npair = len(d)
    grad_ang = (1.0 / npair) * sd
    # d(var)/d ang
    dd = d - mean
    dv = 2.0 / npair * dd
    idx_i = iu[0]
    idx_j = iu[1]
    grad_ang[idx_i, idx_j] += dv
    grad_ang[idx_j, idx_i] += dv
    # d ang / d G = -1/sqrt(1-G^2)
    s = np.sqrt(np.maximum(1e-12, 1.0 - G * G))
    dG = -grad_ang / s
    dG = 0.5 * (dG + dG.T)
    # dG/dpts: G_ij = <p_i,p_j>; dG/dp_i = sum_j dG_ij p_j
    grad = pts @ dG
    # projection to tangent space
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return float(val), grad


def _project(pts):
    return _normalize(pts)


def _lbfgs(pts, max_iter=4000, tol=1e-9):
    """L-BFGS-B on the sphere via tangent projection."""
    x = _normalize(np.asarray(pts, dtype=np.float64)).copy()
    from scipy.optimize import minimize

    def fun(v):
        p = _project(v)
        val, g = _smooth_obj_and_grad(p)
        return val, g

    res = minimize(fun, x, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'gtol': tol, 'maxls': 40})
    return _project(res.x)


def _repulsion_step(pts, step=0.01, iters=300):
    """Vectorized repulsion (inverse-square) on the sphere, fast."""
    pts = _normalize(np.asarray(pts, dtype=np.float64)).copy()
    for _ in range(iters):
        if time.time() > _DEADLINE - 4:
            break
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        ang = np.arccos(dots)
        ang = np.maximum(ang, 0.02)
        rep = 1.0 / (ang * ang)
        np.fill_diagonal(rep, 0.0)
        # force on i = sum_j rep_ij * (p_i - p_j)/|p_i - p_j|
        # vectorized: F_i = sum_j rep_ij p_i - sum_j rep_ij p_j
        F = rep.sum(axis=1)[:, None] * pts - (rep @ pts)
        pts += step * F
        pts = _project(pts)
    return pts


def _active_set_refine(pts, max_iters=80):
    """Move points in the worst facets away from their circumcenter."""
    pts = _normalize(np.asarray(pts, dtype=np.float64)).copy()
    best_cr = _covering_radius_deg(pts)
    step = 0.004
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        simplices = hull.simplices
        v0 = pts[simplices[:, 0]]; v1 = pts[simplices[:, 1]]; v2 = pts[simplices[:, 2]]
        c = v0 + v1 + v2
        cn = np.linalg.norm(c, axis=1)
        valid = cn > 1e-10
        if not np.any(valid):
            break
        cc = c[valid] / cn[valid, None]
        a0 = np.clip(np.einsum('ij,ij->i', cc, v0[valid]), -1, 1)
        a1 = np.clip(np.einsum('ij,ij->i', cc, v1[valid]), -1, 1)
        a2 = np.clip(np.einsum('ij,ij->i', cc, v2[valid]), -1, 1)
        cr = np.max(np.arccos(np.stack([a0, a1, a2], axis=1)), axis=1)
        thr = 0.97 * cr.max()
        worst = np.where(cr > thr)[0]
        if len(worst) == 0:
            break
        # accumulate displacement per point
        disp = np.zeros_like(pts)
        for k in worst:
            tri = simplices[valid][k]
            ck = cc[k]
            for idx in tri:
                disp[idx] -= step * ck
        pts = _project(pts + disp)
        cr_new = _covering_radius_deg(pts)
        if cr_new < best_cr:
            best_cr = cr_new
        else:
            break
    return pts, best_cr


def construct():
    best_pts = None
    best_cr = 180.0

    # diverse initializations
    inits = []
    for s in range(6):
        inits.append(_fibonacci_init(s))
    for s in range(6):
        inits.append(_random_init(s))

    # score initializations (cheap-ish) and keep best few
    scored = []
    for pts in inits:
        if time.time() > _DEADLINE - 70:
            break
        cr = _covering_radius_deg(pts)
        scored.append((cr, pts))
    scored.sort(key=lambda x: x[0])

    # optimize top candidates: fast repulsion -> smooth L-BFGS -> active set
    for cr, pts in scored[:6]:
        if time.time() > _DEADLINE - 12:
            break
        try:
            p1 = _repulsion_step(pts, step=0.008, iters=150)
            p2 = _lbfgs(p1, max_iter=2500)
            p3, cr3 = _active_set_refine(p2, max_iters=60)
            if cr3 < best_cr:
                best_cr = cr3
                best_pts = p3.copy()
        except Exception:
            continue

    # if time remains, polish best a bit more
    if best_pts is not None and time.time() < _DEADLINE - 6:
        try:
            p, cr = _active_set_refine(best_pts, max_iters=40)
            if cr < best_cr:
                best_cr = cr
                best_pts = p.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_init(0)

    best_pts = _normalize(np.asarray(best_pts, dtype=np.float64))
    # de-duplicate coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                rng = np.random.RandomState(42 + i * 1000 + j)
                best_pts[j] = _normalize(best_pts[j] + 1e-5 * rng.randn(3))
    return best_pts
# EVOLVE-BLOCK-END
