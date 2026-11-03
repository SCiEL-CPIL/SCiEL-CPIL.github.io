"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural change: primal-dual interior-point formulation.
Variables: x_i, y_i, r_i for i in 0..n-1.
Constraints (inequality, g_k >= 0):
  - Boundary: x_i - r_i >= 0, y_i - r_i >= 0, 1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
  - Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0  (for i<j)
Objective: maximize sum(r_i)

We use a barrier method with slack variables and solve KKT conditions via Newton steps.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _validate(p):
    """Check disjoint interiors and inside unit square."""
    n = len(p)
    for i in range(n):
        x, y, r = p[i]
        if r < -1e-9:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > 1.0 + 1e-9 or y + r > 1.0 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            d = np.hypot(dx, dy)
            if d < p[i, 2] + p[j, 2] - 1e-8:
                return False
    return True


def _score(p):
    return float(np.sum(p[:, 2]))


def _fix_boundaries(p):
    """Clamp centers so circles stay inside [0,1]^2."""
    for i in range(len(p)):
        r = p[i, 2]
        p[i, 0] = min(max(p[i, 0], r), 1.0 - r)
        p[i, 1] = min(max(p[i, 1], r), 1.0 - r)
    return p


def _initial_hex(n):
    """Hexagonal packing initialization."""
    pts = []
    r = 0.5 / np.sqrt(3)  # hexagonal spacing
    rows = int(np.ceil(n / 3))
    for row in range(rows):
        for col in range(3):
            x = (col + 0.5) / 3.0 + (0.5 / 3.0 if row % 2 else 0)
            y = (row + 0.5) / rows
            if len(pts) < n:
                pts.append((x, y, 0.15))
    return np.array(pts[:n], dtype=float)


def _initial_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:n], dtype=float)


def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    return np.where(x > 0, x + np.log1p(np.exp(-beta * x)) / beta,
                    np.log1p(np.exp(beta * x)) / beta)


def _objective_and_grad(z, beta=50.0):
    """
    z: (3n,) vector [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Returns (objective_value, gradient) for the penalized objective:
      -sum(r_i) - penalty(overlaps) - penalty(boundary violations)
    """
    n = N
    p = z.reshape(n, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(z)

    # Boundary penalties
    for i in range(n):
        for k in range(2):  # x and y
            lo_viol = r[i] - p[i, k]
            hi_viol = p[i, k] + r[i] - 1.0
            if lo_viol > -1.0 / beta:
                obj += _softplus(lo_viol, beta)
                grad[3 * i + k] -= (1.0 if lo_viol > 0 else np.exp(beta * lo_viol)) / beta * (-1)
                grad[3 * i + 2] -= (1.0 if lo_viol > 0 else np.exp(beta * lo_viol)) / beta
            if hi_viol > -1.0 / beta:
                obj += _softplus(hi_viol, beta)
                grad[3 * i + k] += (1.0 if hi_viol > 0 else np.exp(beta * hi_viol)) / beta
                grad[3 * i + 2] += (1.0 if hi_viol > 0 else np.exp(beta * hi_viol)) / beta

    # Objective gradient for -sum(r)
    for i in range(n):
        grad[3 * i + 2] -= 1.0

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            rsum = r[i] + r[j]
            viol = rsum - d
            if viol > -1.0 / beta:
                if d < 1e-12:
                    dx, dy, d = 1e-6, 1e-6, np.hypot(1e-6, 1e-6)
                obj += _softplus(viol, beta)
                w = (1.0 if viol > 0 else np.exp(beta * viol)) / beta
                # d(viol)/d(x_i) = -dx/d, d(viol)/d(y_i) = -dy/d
                grad[3 * i] += w * (-dx / d)
                grad[3 * i + 1] += w * (-dy / d)
                grad[3 * i + 2] += w
                grad[3 * j] += w * (dx / d)
                grad[3 * j + 1] += w * (dy / d)
                grad[3 * j + 2] += w

    return obj, grad


def _physics_relax(p, max_iter=2000, dt=0.05):
    """Physics-based relaxation to resolve overlaps."""
    p = p.copy()
    d = dt
    for it in range(max_iter):
        if _remaining() < 10:
            break
        F = np.zeros((len(p), 2))
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.hypot(dx, dy)
                overlap = p[i, 2] + p[j, 2] - dist
                if overlap > 0:
                    if dist < 1e-12:
                        dx, dy, dist = 1e-6, 1e-6, np.hypot(1e-6, 1e-6)
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    F[i, 0] += fx
                    F[i, 1] += fy
                    F[j, 0] -= fx
                    F[j, 1] -= fy
        for i in range(len(p)):
            r = p[i, 2]
            if p[i, 0] < r:
                F[i, 0] += (r - p[i, 0])
            if p[i, 0] > 1 - r:
                F[i, 0] -= (p[i, 0] - (1 - r))
            if p[i, 1] < r:
                F[i, 1] += (r - p[i, 1])
            if p[i, 1] > 1 - r:
                F[i, 1] -= (p[i, 1] - (1 - r))
        p[:, 0] += dt * F[:, 0]
        p[:, 1] += dt * F[:, 1]
        p = _fix_boundaries(p)
        d *= 0.9995
        # Check convergence
        max_ov = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                ov = p[i, 2] + p[j, 2] - np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                if ov > max_ov:
                    max_ov = ov
        if max_ov < 1e-10:
            break
    return p


def _polish_analytic(p, beta_start=10.0, beta_end=200.0, max_iter=3000):
    """
    Interior-point style polish: iteratively increase barrier parameter and
    solve the penalized problem with L-BFGS-B, keeping feasibility.
    """
    p = p.copy()
    best = p.copy()
    best_score = _score(best)
    n_betas = 8
    betas = np.geomspace(beta_start, beta_end, n_betas)

    for beta in betas:
        if _remaining() < 15:
            break
        z0 = p.flatten().copy()

        def obj_grad(z):
            val, g = _objective_and_grad(z, beta=beta)
            return val, g

        res = minimize(obj_grad, z0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50})

        if res.fun < 0:  # improved
            p_new = res.x.reshape(N, 3)
            p_new = _fix_boundaries(p_new)
            if _validate(p_new):
                sc = _score(p_new)
                if sc > best_score:
                    best_score = sc
                    best = p_new.copy()
                    p = p_new
                else:
                    p = p_new  # still use it as starting point
            else:
                # try to repair
                p_new = _physics_relax(p_new, max_iter=500)
                p_new = _fix_boundaries(p_new)
                if _validate(p_new):
                    sc = _score(p_new)
                    if sc > best_score:
                        best_score = sc
                        best = p_new.copy()
                    p = p_new
    return best


def _grow_radii(p, n_steps=100):
    """Gradually grow radii while maintaining feasibility."""
    p = p.copy()
    best = p.copy()
    best_score = _score(best)
    for s in range(n_steps):
        if _remaining() < 15:
            break
        p[:, 2] *= 1.002
        p = _physics_relax(p, max_iter=300, dt=0.03)
        p = _fix_boundaries(p)
        if _validate(p):
            sc = _score(p)
            if sc > best_score:
                best_score = sc
                best = p.copy()
        else:
            break
    return best


def construct():
    best = _initial_grid(N)
    best_score = _score(best)
    rng = np.random.default_rng(42)

    # Multiple initializations
    inits = []
    inits.append(_initial_grid(N))
    inits.append(_initial_hex(N))
    for _ in range(3):
        p = _initial_grid(N)
        p[:, 0] += rng.uniform(-0.08, 0.08, N)
        p[:, 1] += rng.uniform(-0.08, 0.08, N)
        p = _fix_boundaries(p)
        inits.append(p)

    for p0 in inits:
        if _remaining() < 30:
            break

        # Phase 1: Physics relaxation
        p = _physics_relax(p0.copy(), max_iter=3000, dt=0.05)
        p = _fix_boundaries(p)
        if not _validate(p):
            continue

        # Phase 2: Grow radii
        p = _grow_radii(p, n_steps=80)
        if not _validate(p):
            continue

        # Phase 3: Analytic polish with increasing barrier
        p = _polish_analytic(p)
        if not _validate(p):
            continue

        sc = _score(p)
        if sc > best_score:
            best_score = sc
            best = p.copy()

    # Final polish on best
    if _remaining() > 20:
        best = _polish_analytic(best.copy())
        if not _validate(best):
            # fallback
            best = _fix_boundaries(best)

    return best
