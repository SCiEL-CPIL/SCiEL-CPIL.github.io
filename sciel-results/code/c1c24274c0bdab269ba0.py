"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _compute_radius(pts):
    """Return the packing radius for a set of centres (larger is better)."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: distance to nearest edge of [-1/2, 1/2]^2
    d = 0.5 - np.max(np.abs(pts), axis=1)
    r = float(d.min())
    if r <= 0.0:
        return 0.0
    # pairwise
    for i in range(N):
        diff = pts[i + 1:] - pts[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        m = dist.min()
        if m < 2.0 * r:
            r = 0.5 * float(m)
    return r


def _slsqp_solve(x0, deadline):
    """Run SLSQP to maximise the minimum clearance. Returns improved pts or None."""
    n = N
    # pairwise index pairs
    iu = np.triu_indices(n, k=1)
    pi, pj = iu[0], iu[1]
    npair = len(pi)

    def obj(v):
        # constant objective (we maximise clearance via constraints only)
        return 0.0

    def obj_grad(v):
        return np.zeros(2 * n)

    def cons(v):
        x = v[:2 * n].reshape(n, 2)
        # pairwise: |xi - xj| - 2*t >= 0
        diff = x[pi] - x[pj]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        pair = d - 2.0 * v[2 * n]
        # boundary: 0.5 - |xk| - t >= 0
        bnd = 0.5 - np.max(np.abs(x), axis=1) - v[2 * n]
        return np.concatenate([pair, bnd])

    def cons_jac(v):
        x = v[:2 * n].reshape(n, 2)
        t = v[2 * n]
        # pairwise rows
        diff = x[pi] - x[pj]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        # guard
        d_safe = np.where(d > 1e-12, d, 1e-12)
        inv = 1.0 / d_safe
        J = np.zeros((npair + 4 * n, 2 * n + 1))
        # For each pair (pi, pj): d/dxi diff = +1, d/dxj diff = -1
        # row for pair k: J[k, 2*pi] = diff[0]*inv, J[k, 2*pi+1]=diff[1]*inv,
        # J[k, 2*pj] = -diff[0]*inv, J[k, 2*pj+1] = -diff[1]*inv, J[k, 2*n] = -2
        cols_i = 2 * pi
        cols_j = 2 * pj
        J[np.arange(npair), cols_i] = diff[:, 0] * inv
        J[np.arange(npair), cols_i + 1] = diff[:, 1] * inv
        J[np.arange(npair), cols_j] = -diff[:, 0] * inv
        J[np.arange(npair), cols_j + 1] = -diff[:, 1] * inv
        J[np.arange(npair), 2 * n] = -2.0
        # boundary rows: 0.5 - max(|x|) - t
        # max over components; subgradient: pick argmax component
        base = npair
        for k in range(n):
            ax = np.abs(x[k])
            am = ax.max()
            # subgradient: for each component, 1 if it's the max (and positive)
            # d/dx of |x| is sign(x) when x != 0; max picks the dominant one
            row = base + k
            # gradient wrt x[k,0]: -sign if comp0 is the max
            if ax[0] >= ax[1]:
                J[row, 2 * k] = -np.sign(x[k, 0]) if x[k, 0] != 0 else 0.0
            else:
                J[row, 2 * k + 1] = -np.sign(x[k, 1]) if x[k, 1] != 0 else 0.0
            J[row, 2 * n] = -1.0
        # rows for other 3 boundary constraints? We used 4*n but only n rows.
        # Actually boundary has 4 sides but we folded into max over abs => n rows.
        # So total rows = npair + n. Fix J shape.
        # (We'll recompute J with correct size below to be safe.)
        return J

    # Recompute a clean Jacobian with correct row count (npair + n)
    def cons_jac2(v):
        x = v[:2 * n].reshape(n, 2)
        diff = x[pi] - x[pj]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d_safe = np.where(d > 1e-12, d, 1e-12)
        inv = 1.0 / d_safe
        J = np.zeros((npair + n, 2 * n + 1))
        cols_i = 2 * pi
        cols_j = 2 * pj
        J[np.arange(npair), cols_i] = diff[:, 0] * inv
        J[np.arange(npair), cols_i + 1] = diff[:, 1] * inv
        J[np.arange(npair), cols_j] = -diff[:, 0] * inv
        J[np.arange(npair), cols_j + 1] = -diff[:, 1] * inv
        J[np.arange(npair), 2 * n] = -2.0
        base = npair
        for k in range(n):
            ax = np.abs(x[k])
            row = base + k
            if ax[0] >= ax[1]:
                J[row, 2 * k] = -np.sign(x[k, 0]) if x[k, 0] != 0 else 0.0
            else:
                J[row, 2 * k + 1] = -np.sign(x[k, 1]) if x[k, 1] != 0 else 0.0
            J[row, 2 * n] = -1.0
        return J

    def con(v):
        x = v[:2 * n].reshape(n, 2)
        diff = x[pi] - x[pj]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        pair = d - 2.0 * v[2 * n]
        bnd = 0.5 - np.max(np.abs(x), axis=1) - v[2 * n]
        return np.concatenate([pair, bnd])

    constraints = [{'type': 'ineq', 'fun': con, 'jac': cons_jac2}]

    v0 = np.concatenate([x0.ravel(), [0.0]])
    # bounds: x in [-0.5, 0.5], t in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.5)]

    try:
        res = minimize(obj, v0, method='SLSQP', jac=obj_grad,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 400, 'ftol': 1e-10, 'disp': False})
    except Exception:
        return None
    if res.x is None:
        return None
    pts = res.x[:2 * n].reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return None
    return pts


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s

    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best = grid[:N].copy()
    best_r = _compute_radius(best)

    # Try SLSQP from the grid
    if time.time() < deadline - 30.0:
        cand = _slsqp_solve(best.copy(), deadline)
        if cand is not None:
            r = _compute_radius(cand)
            if r > best_r:
                best = cand
                best_r = r

    # A few small jittered restarts if time remains
    rng = np.random.default_rng(12345)
    for attempt in range(3):
        if time.time() > deadline - 40.0:
            break
        x0 = best.copy()
        x0 = x0 + rng.normal(0, 1e-3, size=x0.shape)
        # clamp to be safely inside
        x0 = np.clip(x0, -0.49, 0.49)
        cand = _slsqp_solve(x0, deadline)
        if cand is not None:
            r = _compute_radius(cand)
            if r > best_r:
                best = cand
                best_r = r

    return best
# EVOLVE-BLOCK-END
