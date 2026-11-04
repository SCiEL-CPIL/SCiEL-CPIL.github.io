"""Circle packing in perimeter-4 rectangle.

Different modelling family from the seed:
  - radius is a first-class variable (not derived from geometry after the fact)
  - hard constraints (walls + pairwise separation) via SLSQP
  - physics-inspired repulsion relaxation as a robust pre-conditioner
  - finer aspect-ratio sweep + multi-start
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
START_TIME = time.time()
DEADLINE = START_TIME + 112.0  # safety margin well under the 150 s wall-clock limit


def _elapsed():
    return time.time() - START_TIME


def _check_validity(centers, w, h, r):
    """Return True if all circles fit inside the box and are pairwise disjoint."""
    if r <= 0:
        return False
    if np.any(centers[:, 0] < -1e-7) or np.any(centers[:, 0] > w + 1e-7):
        return False
    if np.any(centers[:, 1] < -1e-7) or np.any(centers[:, 1] > h + 1e-7):
        return False
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= 2.0 * r - 1e-7


def _total_radius(centers, w, h, r):
    """Objective value: N * r, only if the packing is valid."""
    if not _check_validity(centers, w, h, r):
        return 0.0
    return N * r


def _separate(centers, w, h, r, iters=300, lr=0.02, wall=0.5):
    """Physics-inspired repulsion relaxation.

    Pushes overlapping circles apart and keeps them inside the box.
    Returns a (possibly improved) valid configuration of radius r.
    """
    centers = centers.astype(float).copy()
    n = centers.shape[0]
    for _ in range(iters):
        # pairwise repulsion
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        forces = np.zeros_like(centers)
        min_dist = 2.0 * r
        # only act on pairs that overlap (or are very close)
        mask = dist < min_dist * 1.05
        for i in range(n):
            for j in range(i + 1, n):
                if not mask[i, j]:
                    continue
                d = diff[i, j]
                if dist[i, j] < 1e-9:
                    d = np.array([1e-4, 1e-4])
                    dist[i, j] = 1e-4
                overlap = min_dist - dist[i, j]
                if overlap > 0:
                    push = (overlap / dist[i, j]) * 0.5
                    forces[i] -= push * d
                    forces[j] += push * d
        centers += lr * forces
        # wall repulsion
        for i in range(n):
            if centers[i, 0] < r:
                centers[i, 0] += wall * (r - centers[i, 0])
            elif centers[i, 0] > w - r:
                centers[i, 0] -= wall * (centers[i, 0] - (w - r))
            if centers[i, 1] < r:
                centers[i, 1] += wall * (r - centers[i, 1])
            elif centers[i, 1] > h - r:
                centers[i, 1] -= wall * (centers[i, 1] - (h - r))
        # clamp inside
        centers[:, 0] = np.clip(centers[:, 0], r * 0.5, w - r * 0.5)
        centers[:, 1] = np.clip(centers[:, 1], r * 0.5, h - r * 0.5)
    return centers


def _hex_initialization(w, h, seed=0):
    """Hexagonal-ish grid initialization inside the box."""
    rng = np.random.default_rng(seed)
    # estimate grid
    area = w * h
    # rough radius guess
    r0 = np.sqrt(area / (N * np.pi))
    spacing = 2.2 * r0
    cols = max(1, int(np.ceil(w / spacing)))
    rows = max(1, int(np.ceil(h / spacing)))
    centers = []
    for ri in range(rows):
        for ci in range(cols):
            x = (ci + 0.5) * spacing
            y = (ri + 0.5) * spacing
            if ri % 2 == 1:
                x += 0.5 * spacing
            if len(centers) < N:
                centers.append([x, y])
    while len(centers) < N:
        centers.append([w / 2, h / 2])
    centers = np.array(centers[:N], dtype=float)
    # small jitter
    centers += rng.uniform(-spacing * 0.1, spacing * 0.1, size=centers.shape)
    return centers


def _grid_initialization(w, h, seed=0):
    rng = np.random.default_rng(seed)
    cols = int(np.sqrt(N * w / h))
    cols = max(1, cols)
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.15, 0.85, cols) * w
    ys = np.linspace(0.15, 0.85, rows) * h
    centers = []
    for i in range(N):
        ri = i // cols
        ci = i % cols
        centers.append([xs[ci], ys[ri]])
    centers = np.array(centers, dtype=float)
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    return centers


def _random_initialization(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=(N, 2)) * np.array([w, h])


def _slsqp_optimize(centers0, w, h, r0, maxiter=200):
    """Constrained SLSQP: variables = 2N centers + 1 radius. Maximize r."""
    x0 = np.concatenate([centers0.flatten(), [r0]])

    def neg_r(x):
        return -x[-1]

    cons = []
    # wall constraints (each circle: center - r >= 0 and (w - center) - r >= 0)
    for i in range(N):
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: x[2 * i] - x[-1],
                     'jac': lambda x, i=i: np.concatenate([np.eye(2 * N + 1)[2 * i], [-1.0]])})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: (w - x[2 * i]) - x[-1],
                     'jac': lambda x, i=i: np.concatenate([-np.eye(2 * N + 1)[2 * i], [-1.0]])})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: x[2 * i + 1] - x[-1],
                     'jac': lambda x, i=i: np.concatenate([np.eye(2 * N + 1)[2 * i + 1], [-1.0]])})
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i: (h - x[2 * i + 1]) - x[-1],
                     'jac': lambda x, i=i: np.concatenate([-np.eye(2 * N + 1)[2 * i + 1], [-1.0]])})
    # pairwise separation: ||c_i - c_j|| - 2r >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def _sep(x, i=i, j=j):
                di = x[2 * i:2 * i + 2]
                dj = x[2 * j:2 * j + 2]
                d = np.linalg.norm(di - dj)
                return d - 2.0 * x[-1]

            def _sep_jac(x, i=i, j=j):
                di = x[2 * i:2 * i + 2]
                dj = x[2 * j:2 * j + 2]
                diff = di - dj
                d = np.linalg.norm(diff)
                if d < 1e-9:
                    grad_i = np.array([1e-6, 0.0])
                else:
                    grad_i = diff / d
                jac = np.zeros(2 * N + 1)
                jac[2 * i:2 * i + 2] = grad_i
                jac[2 * j:2 * j + 2] = -grad_i
                jac[-1] = -2.0
                return jac

            cons.append({'type': 'ineq', 'fun': _sep, 'jac': _sep_jac})

    # radius lower bound handled via bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, w - 1e-6))
        bounds.append((1e-6, h - 1e-6))
    bounds.append((1e-4, None))

    try:
        res = minimize(neg_r, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-8})
        if res.success or res.fun < -r0 * 0.999:
            xc = res.x[:2 * N].reshape(N, 2)
            rc = res.x[-1]
            if _check_validity(xc, w, h, rc):
                return xc, rc
    except Exception:
        pass
    return None, 0.0


def _solve_aspect(aspect, maxiter=150, n_starts=2, seed_base=0):
    """Solve for a fixed aspect ratio with multiple starts."""
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    if w <= 0 or h <= 0:
        return None, 0.0, w, h

    best_centers = None
    best_r = 0.0

    inits = [
        _hex_initialization(w, h, seed=seed_base),
        _grid_initialization(w, h, seed=seed_base + 1),
    ]
    if n_starts > 2:
        inits.append(_random_initialization(w, h, seed=seed_base + 2))

    for k, c0 in enumerate(inits):
        if _elapsed() > DEADLINE - 4.0:
            break
        # pre-condition with a modest radius guess
        r_guess = min(w, h) / (2.0 * (int(np.ceil(np.sqrt(N))) + 1)) * 1.2
        r_guess = max(0.02, min(r_guess, min(w, h) / 2.0))
        c_sep = _separate(c0, w, h, r_guess, iters=120, lr=0.02)
        # then constrained optimization
        c_opt, r_opt = _slsqp_optimize(c_sep, w, h, r_guess, maxiter=maxiter)
        if c_opt is not None and r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt
        # also keep the relaxed one if it's valid and better
        if _check_validity(c_sep, w, h, r_guess) and r_guess > best_r:
            best_r = r_guess
            best_centers = c_sep

    return best_centers, best_r, w, h


def construct():
    global DEADLINE
    best_val = 0.0
    best_centers = None
    best_r = 0.0
    best_w = 1.0
    best_h = 1.0

    # finer aspect sweep: from near-square to elongated
    aspects = np.concatenate([
        np.linspace(0.55, 1.0, 8),
        np.linspace(1.0, 1.6, 8),
    ])
    aspects = np.unique(np.round(aspects, 4))

    for idx, aspect in enumerate(aspects):
        if _elapsed() > DEADLINE - 6.0:
            break
        c, r, w, h = _solve_aspect(aspect, maxiter=120, n_starts=2, seed_base=idx)
        if c is not None and r > 0:
            val = _total_radius(c, w, h, r)
            if val > best_val:
                best_val = val
                best_r = r
                best_centers = c
                best_w = w
                best_h = h

    # Fallback if nothing valid
    if best_centers is None:
        w, h = 1.0, 1.0
        c = _grid_initialization(w, h, seed=0)
        r = 0.03
        c = _separate(c, w, h, r, iters=200)
        # find max r that is valid via bisection
        lo, hi = 0.01, min(w, h) / 2.0
        for _ in range(20):
            mid = (lo + hi) / 2
            cc = _separate(_grid_initialization(w, h, seed=0), w, h, mid, iters=200)
            if _check_validity(cc, w, h, mid):
                lo = mid
                best_centers = cc
                best_r = mid
            else:
                hi = mid
        best_w, best_h = w, h
        best_val = _total_radius(best_centers, w, h, best_r)

    out = np.zeros((N, 3))
    out[:, 0] = best_centers[:, 0]
    out[:, 1] = best_centers[:, 1]
    out[:, 2] = best_r
    return out


if __name__ == "__main__":
    arr = construct()
    print(arr.shape, arr[:3])
