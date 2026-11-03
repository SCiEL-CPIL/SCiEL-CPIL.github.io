import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 12.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _energy_grad(pts):
    """Energy and gradient via chordal identity d^2 = 2 - 2(x_i . x_j)."""
    n = pts.shape[0]
    G = pts @ pts.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(np.maximum(D2, 1e-30))
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(D[iu])))
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    # dE/dx_i = sum_j W_ij (x_j - x_i)
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    # project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad

def _precise_energy(pts):
    n = pts.shape[0]
    terms = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            d = math.sqrt(max(float(np.dot(pi - pts[j], pi - pts[j])), 1e-30))
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)

def _sobol_like(n, seed=7):
    rng = np.random.default_rng(seed)
    # quasi-random via stratified low-discrepancy construction
    pts = []
    # use a golden-angle spiral with a different parameter for diversity
    for k in range(n):
        t = (k + 0.5) / n
        z = 1.0 - 2.0 * t
        r = np.sqrt(max(0.0, 1.0 - z * z))
        a = 2.0 * np.pi * (k * 0.61803398875)
        pts.append([r * np.cos(a), r * np.sin(a), z])
    return np.array(pts)

def _riemannian_lbfgs(pts, max_iter=2000, ftol=1e-16, gtol=1e-14, time_budget=30.0):
    """L-BFGS-B directly in tangent coordinates (Euclidean approx of the manifold)."""
    n = pts.shape[0]
    t0 = time.time()
    # build an orthonormal tangent basis per point
    basis = np.zeros((n, 3, 3))
    for i in range(n):
        p = pts[i]
        e1 = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1 = e1 - np.dot(e1, p) * p
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(p, u1)
        u2 = u2 / np.linalg.norm(u2)
        basis[i] = np.stack([u1, u2], axis=1)

    def obj_grad(flat):
        if time.time() - t0 > time_budget:
            raise RuntimeError("timeout")
        u = flat.reshape(n, 2)
        # map tangent coords back to sphere
        p = pts + (u[:, 0:1] * basis[:, :, 0] + u[:, 1:2] * basis[:, :, 1])
        p = _project(p)
        e, g = _energy_grad(p)
        # pull gradient back to tangent coords
        gu = np.einsum('nij,nj->ni', basis, g)
        return e, gu.reshape(-1)

    try:
        res = minimize(obj_grad, np.zeros((n, 2)), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'maxfun': max_iter * 8,
                                'ftol': ftol, 'gtol': gtol, 'maxls': 40})
        u = res.x.reshape(n, 2)
        p = pts + (u[:, 0:1] * basis[:, :, 0] + u[:, 1:2] * basis[:, :, 1])
        return _project(p)
    except Exception:
        return pts

def _polish(pts, max_iter=8000, tol=1e-15, time_budget=40.0):
    """Adaptive-step Riemannian gradient descent for final polish."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _precise_energy(pts)
    step = 0.3
    t0 = time.time()
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        e, g = _energy_grad(best_pts)
        if np.linalg.norm(g) < tol:
            break
        cand = _project(best_pts - step * g)
        ce = _precise_energy(cand)
        if ce < best_e:
            best_e = ce
            best_pts = cand
            step = min(step * 1.25, 1.0)
        else:
            ok = False
            for _ in range(12):
                step *= 0.5
                cand = _project(best_pts - step * g)
                ce = _precise_energy(cand)
                if ce < best_e:
                    best_e = ce
                    best_pts = cand
                    ok = True
                    break
            if not ok and step < 1e-11:
                break
    return best_pts

def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    G = X @ X.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, np.inf)
    if np.min(D2) < (1e-9) ** 2:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + (_DEADLINE - _SAFETY)
    rng = np.random.default_rng(42)

    initializations = []
    for off in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=off))
    for s in range(8):
        initializations.append(_random_sphere(np.random.default_rng(200 + s)))
    initializations.append(_sobol_like(N, seed=3))

    best_pts = None
    best_energy = float('inf')

    scored = []
    for init in initializations:
        if time.time() > deadline - 40.0:
            break
        scored.append((_precise_energy(init), init))
    scored.sort(key=lambda x: x[0])

    # optimize top candidates with Riemannian L-BFGS-B
    for e0, init in scored[:6]:
        if time.time() > deadline - 20.0:
            break
        opt = _riemannian_lbfgs(init, max_iter=1500, ftol=1e-15, gtol=1e-12, time_budget=18.0)
        e = _precise_energy(opt)
        if e < best_energy:
            best_energy = e
            best_pts = opt

    if best_pts is None:
        best_pts = _project(_fibonacci_spiral(N, 0.5))
        best_energy = _precise_energy(best_pts)

    # basin hopping: kick + optimize, accept improvements
    cur_pts = best_pts.copy()
    cur_e = best_energy
    kick = 0.4
    for hop in range(40):
        if time.time() > deadline - 12.0:
            break
        kicked = _project(cur_pts + kick * rng.standard_normal((N, 3)))
        opt = _riemannian_lbfgs(kicked, max_iter=800, ftol=1e-13, gtol=1e-10, time_budget=10.0)
        e = _precise_energy(opt)
        if e < cur_e:
            cur_e = e
            cur_pts = opt
            if e < best_energy:
                best_energy = e
                best_pts = opt
            kick = max(kick * 0.92, 0.08)
        else:
            kick = max(kick * 0.98, 0.08)

    # final high-precision polish
    if time.time() < deadline - 6.0:
        pol = _polish(best_pts, max_iter=8000, tol=1e-15, time_budget=30.0)
        e = _precise_energy(pol)
        if e < best_energy:
            best_energy = e
            best_pts = pol

    # one last tight Riemannian solve
    if time.time() < deadline - 2.0:
        fin = _riemannian_lbfgs(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-15, time_budget=12.0)
        e = _precise_energy(fin)
        if e < best_energy:
            best_energy = e
            best_pts = fin

    if not _valid(best_pts):
        best_pts = _project(_fibonacci_spiral(N, 0.5))
    return best_pts
