"""Smooth inverse-power-sum packing: hex seed + multi-restart L-BFGS-B polish.

Structural idea (from inspiration): minimize a smooth sum of inverse powers of
pairwise and wall distances.  This is differentiable everywhere (coincident
points handled by an epsilon floor), so L-BFGS-B gets clean gradients and
drives the configuration toward maximizing the minimum separation.

Refinement machinery (from current program): coherent hexagonal seed,
multi-restart with shrinking perturbation, two-phase power schedule, and a
final high-precision polish, all under an explicit deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 1234567


def _radius_of(c):
    """True packing radius for centers c (N,2)."""
    c = np.asarray(c)
    wall = np.min(HALF - np.abs(c))
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))


def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers.

    Selects the n points closest to the centre so the packing is symmetric and
    boundary-aware (drops the outermost, least-well-supported points).
    """
    for attempt in range(200):
        r0 = 0.036
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            x_start = x0 + (h / 2 if i % 2 else 0.0)
            if x_start > HALF - r0:
                continue
            n_cols = int((HALF - r0 - x_start) / h) + 1
            for j in range(n_cols):
                x = x_start + j * h
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            if len(pts) == n:
                return pts
            # keep the n closest to centre for a symmetric, well-supported set
            dctr = np.linalg.norm(pts, axis=1)
            order = np.argsort(dctr)
            return pts[order[:n]]
        r0 *= 0.995
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _energy_and_grad(x, p, eps):
    """Smooth sum of inverse powers of pairwise and wall distances.

    Returns (energy, grad) where x is a flat (2N,) vector of centers.
    """
    c = x.reshape(N, 2)

    # --- pairwise ---
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dclip = np.maximum(dist, eps)
    energy_pairs = np.sum(np.power(dclip, -p))

    # gradient of sum_{j!=i} d_ij^-p wrt c_i = -p * sum_j d_ij^(-p-2) * (c_i - c_j)
    weights = -p * np.power(dclip, -p - 2)
    np.fill_diagonal(weights, 0.0)
    sum_w = np.sum(weights, axis=1)
    weighted_c = weights @ c
    grads = c * sum_w[:, None] - weighted_c

    # --- walls ---
    d_left = HALF + c[:, 0]
    d_right = HALF - c[:, 0]
    d_bottom = HALF + c[:, 1]
    d_top = HALF - c[:, 1]
    w = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
    wclip = np.maximum(w, eps)
    energy_walls = np.sum(np.power(wclip, -p))

    # gradient of w_i^-p wrt coords: -p w_i^(-p-1) * d(w_i)/d(coord)
    coeff = -p * np.power(wclip, -p - 1)
    tol = 1e-9
    gx = np.zeros(N)
    gy = np.zeros(N)
    gx += (np.abs(d_left - w) <= tol) * 1.0
    gx += (np.abs(d_right - w) <= tol) * (-1.0)
    gy += (np.abs(d_bottom - w) <= tol) * 1.0
    gy += (np.abs(d_top - w) <= tol) * (-1.0)
    grads[:, 0] += coeff * gx
    grads[:, 1] += coeff * gy

    return energy_pairs + energy_walls, grads.ravel()


def _optimize_energy(c0, p, eps, rem):
    """Run L-BFGS-B on the inverse-power objective; return (centers, radius)."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)

    def obj(x):
        e, _ = _energy_and_grad(x, p, eps)
        return e

    def grad(x):
        _, g = _energy_and_grad(x, p, eps)
        return g

    best_x = x0
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50})
        if res.x is not None and np.all(np.isfinite(res.x)):
            # accept if it improves the objective (or just succeeded)
            if res.fun < obj(x0) or res.success:
                best_x = res.x
    except Exception:
        pass

    best_c = np.clip(best_x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    r = _radius_of(best_c)
    return best_c, r


def construct():
    t_start = time.time()
    deadline = 115.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    eps = 1e-5

    # ---- Phase A: coarse spread (moderate power) ----
    # A few restarts with shrinking perturbation to find a good basin.
    coarse_restarts = 0
    while time.time() - t_start < deadline - 25.0:
        coarse_restarts += 1
        scale = 0.02 / (1.0 + 0.08 * coarse_restarts)
        c_start = best_c + rng.normal(0.0, scale, size=best_c.shape)
        c_start = np.clip(c_start, -HALF + 1e-6, HALF - 1e-6)
        # reject near-coincident starts
        d = c_start[:, None, :] - c_start[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-4:
            c_start = best_c.copy()
        rem = deadline - (time.time() - t_start)
        if rem < 12.0:
            break
        c_opt, r_opt = _optimize_energy(c_start, p=4.0, eps=eps, rem=rem)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    # ---- Phase B: fine polish (large power sharpens the min-distance) ----
    rem = deadline - (time.time() - t_start)
    if rem > 20.0:
        c_polish, r_polish = _optimize_energy(best_c, p=12.0, eps=1e-6, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # ---- Phase C: ultra-fine final polish if time remains ----
    rem = deadline - (time.time() - t_start)
    if rem > 15.0:
        c_polish, r_polish = _optimize_energy(best_c, p=20.0, eps=1e-7, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # final safety
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
