"""Logarithmic energy minimization for N points on the unit sphere.
Multi-start optimization: Sobol/Fibonacci/random initializations, a
stochastic Riemannian (Langevin) global-search phase to escape local minima,
L-BFGS-B refinement, and a final high-precision polish pass.

Energy:  E = sum_{i<j} log(1/d_ij)  with chordal distances d_ij.
Stable form:  d = 2 sin(theta/2), theta = 2 arcsin(d/2)
             E = -sum_{i<j} log(2 sin(theta/2))
Gradient (tangent-projected) derived from the pairwise potential.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0          # safety margin from 150s wall-clock limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _pair_theta(pts):
    """Pairwise angles theta_ij in (0, pi).  Returns full NxN matrix."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.clip(d2, 0.0, 4.0))
    # theta = 2 arcsin(d/2); clip to avoid domain issues
    theta = 2.0 * np.arcsin(np.clip(0.5 * d, 0.0, 1.0))
    return theta


def _energy(pts):
    """Fast energy: -sum_{i<j} log(2 sin(theta_ij/2))."""
    theta = _pair_theta(pts)
    s = np.sin(0.5 * theta)
    # avoid log(0) on diagonal / near-collinear
    s = np.clip(s, 1e-15, 1.0)
    val = -np.log(2.0 * s)
    iu = np.triu_indices(N, k=1)
    return float(np.sum(val[iu]))


def _energy_grad(pts):
    """Energy and tangent-projected gradient.

    Pairwise potential  phi(theta) = -log(2 sin(theta/2)),  phi'(theta) = -0.5 cot(theta/2).
    dE/dx_i (raw) = - sum_{j != i} phi'(theta_ij) * (x_i - x_j)/d_ij
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 0.0)
    d = np.sqrt(np.clip(d2, 1e-30, 4.0))
    theta = 2.0 * np.arcsin(np.clip(0.5 * d, 0.0, 1.0))

    s = np.sin(0.5 * theta)
    c = np.cos(0.5 * theta)
    s = np.clip(s, 1e-15, 1.0)
    val = -np.log(2.0 * s)
    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(val[iu]))

    # phi'(theta) = -0.5 cot(theta/2) = -0.5 * c/s
    dphi = -0.5 * c / s
    np.fill_diagonal(dphi, 0.0)
    # weight w_ij = -phi'(theta_ij)/d_ij  (finite; diagonal 0)
    W = dphi / np.maximum(d, 1e-15)
    np.fill_diagonal(W, 0.0)

    # dE/dx_i = - sum_j W_ij (x_i - x_j) = -[x_i sum_j W_ij - sum_j W_ij x_j]
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx          # = sum_j W_ij (x_i - x_j)
    grad = -raw_grad

    # project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _precise_energy(pts):
    """Energy using math.fsum for accuracy (chordal distances)."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
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


def _sobol_sphere(n, seed=0):
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _optimize(pts, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': maxls,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _riemannian_perturb(pts, scale=0.05, rng=None):
    """Perturb points in tangent space to escape local minima."""
    if rng is None:
        rng = np.random.default_rng()
    n = pts.shape[0]
    random_vecs = rng.standard_normal((n, 3))
    dot = np.sum(random_vecs * pts, axis=1, keepdims=True)
    tangent_vecs = random_vecs - dot * pts
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    tangent_vecs = tangent_vecs / norms
    perturbed = pts + scale * tangent_vecs
    return _project(perturbed)


def _langevin(pts, rng, T0=0.30, Tend=0.02, steps=400, dt=0.05,
              time_budget=25.0):
    """Stochastic Riemannian (Langevin) dynamics with temperature annealing.

    Escapes the local minima that trap single-start gradient descent.
    Returns the best configuration encountered (by energy).
    """
    t = time.time()
    cur = pts.copy()
    best = cur.copy()
    best_e = _energy(cur)
    n = N
    for k in range(steps):
        if time.time() - t > time_budget:
            break
        frac = k / max(1, steps - 1)
        T = T0 * (Tend / T0) ** frac
        e, g = _energy_grad(cur)
        noise = rng.standard_normal((n, 3))
        dot = np.sum(noise * cur, axis=1, keepdims=True)
        noise = noise - dot * cur
        nxt = cur - dt * g + math.sqrt(2.0 * dt * T) * noise
        cur = _project(nxt)
        e_new = _energy(cur)
        if e_new < best_e:
            best_e = e_new
            best = cur.copy()
    return best, best_e


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0   # 10s safety margin

    rng = np.random.default_rng(42)

    # ---- Generate diverse initializations ---------------------------------
    initializations = []
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.03 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))

    # ---- Score initializations (cheap energy) -----------------------------
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    def consider(candidate):
        nonlocal best_pts, best_energy
        e = _precise_energy(candidate)
        if e < best_energy:
            best_energy = e
            best_pts = candidate.copy()
            return True
        return False

    top_candidates = scored_inits[:30]

    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 12.0:
            break

        # Phase A: coarse L-BFGS-B
        opt = _optimize(init_pts, max_iter=400, ftol=1e-10, gtol=1e-6, maxls=30)
        consider(opt)

        # Phase B: stochastic global search (Langevin) to escape local minima
        if time.time() < deadline - 12.0:
            lang, _ = _langevin(opt, rng, T0=0.30, Tend=0.02,
                                steps=300, dt=0.05, time_budget=12.0)
            consider(lang)
            # refine the best Langevin point
            if time.time() < deadline - 10.0:
                opt2 = _optimize(lang, max_iter=800, ftol=1e-13, gtol=1e-10, maxls=40)
                consider(opt2)

        # Phase C: fine L-BFGS-B on the current best
        if time.time() < deadline - 8.0:
            opt3 = _optimize(opt, max_iter=1500, ftol=1e-15, gtol=1e-12, maxls=50)
            consider(opt3)

        # Phase D: Riemannian perturbation restarts for top candidates
        if idx < 5 and time.time() < deadline - 10.0:
            for perturb_scale in [0.02, 0.05, 0.1]:
                if time.time() > deadline - 6.0:
                    break
                perturbed = _riemannian_perturb(opt3, scale=perturb_scale, rng=rng)
                popt = _optimize(perturbed, max_iter=1200, ftol=1e-14,
                                 gtol=1e-11, maxls=50)
                consider(popt)

    # ---- Final high-precision polish --------------------------------------
    if best_pts is not None and time.time() < deadline - 4.0:
        refined = _optimize(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-13, maxls=50)
        consider(refined)

        # Deterministic micro-perturbation sweep (avoids RNG cost, cheap)
        if time.time() < deadline - 2.0:
            for perturb_scale in [0.01, 0.02, 0.03]:
                if time.time() > deadline - 1.0:
                    break
                perturbed = _riemannian_perturb(refined, scale=perturb_scale, rng=rng)
                popt = _optimize(perturbed, max_iter=1500, ftol=1e-16,
                                 gtol=1e-14, maxls=50)
                consider(popt)

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)

    best_pts = _project(best_pts)

    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy) or not _valid(best_pts):
        best_pts = _project(_fibonacci_spiral(N, offset=0.5))

    return best_pts
