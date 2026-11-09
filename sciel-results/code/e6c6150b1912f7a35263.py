"""Optimized: L-BFGS-B on smooth log-sum with diameter penalty, many restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _true_score(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    diam = d.max()
    if diam < 1e-15:
        return 0.0
    scaled = pts * (2.0 / diam)
    d2 = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d2, 0.0)
    mask = ~np.eye(n, dtype=bool)
    dists = d2[mask]
    if np.any(dists < 1e-15):
        return 0.0
    log_delta = np.sum(np.log(dists))
    score = np.exp(log_delta) / (8.0 ** 8)
    return float(score)

def _obj_and_grad(params, alpha=5.0, lam=50.0):
    pts = params.reshape(8, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, 1e-12)
    
    iu = np.triu_indices(8, k=1)
    pair_d = d[iu]
    log_sum = np.sum(np.log(pair_d))
    
    diam = d.max()
    pen = lam * max(0.0, diam - 2.0) ** 2
    
    # Objective to minimize: -alpha * 2 * log_sum + pen
    obj = -alpha * 2.0 * log_sum + pen
    
    # Gradient
    grad = np.zeros((8, 2))
    for i, j in zip(*iu):
        diff_ij = pts[i] - pts[j]
        d2 = float(np.dot(diff_ij, diff_ij))
        if d2 < 1e-20:
            continue
        g = diff_ij / d2
        grad[i] += g
        grad[j] -= g
    
    if diam > 2.0:
        ii, jj = np.unravel_index(np.argmax(d), d.shape)
        diff_ij = pts[ii] - pts[jj]
        dn = float(np.linalg.norm(diff_ij))
        if dn > 1e-15:
            pg = 2.0 * lam * (diam - 2.0) * (diff_ij / dn)
            grad[ii] += pg
            grad[jj] -= pg
    
    grad_flat = (-alpha * 2.0 * grad).flatten()
    return obj, grad_flat

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    # Regular octagon
    angles = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _true_score(octagon)
    best_pts = octagon.copy()
    
    rng = np.random.default_rng(12345)
    
    # Restart initializations
    def make_init(r):
        if r % 5 == 0:
            return octagon.flatten() + rng.normal(0, 0.05, 16)
        elif r % 5 == 1:
            return octagon.flatten() + rng.normal(0, 0.2, 16)
        elif r % 5 == 2:
            return rng.uniform(-1, 1, 16)
        elif r % 5 == 3:
            # Perturbed regular polygon
            a = 2 * np.pi * np.arange(8) / 8 + rng.normal(0, 0.3, 8)
            rad = 1.0 + rng.normal(0, 0.2, 8)
            return np.column_stack([rad * np.cos(a), rad * np.sin(a)]).flatten()
        else:
            # Two rings
            a1 = 2 * np.pi * np.arange(4) / 4 + rng.normal(0, 0.2, 4)
            a2 = 2 * np.pi * np.arange(4) / 4 + np.pi/4 + rng.normal(0, 0.2, 4)
            p1 = np.column_stack([0.5*np.cos(a1), 0.5*np.sin(a1)])
            p2 = np.column_stack([1.0*np.cos(a2), 1.0*np.sin(a2)])
            return np.vstack([p1, p2]).flatten()
    
    alphas = [1.0, 3.0, 10.0, 30.0]
    
    restart = 0
    while time.time() < deadline - 8.0:
        x0 = make_init(restart)
        for alpha in alphas:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    _obj_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    args=(alpha, 50.0),
                    options={'maxiter': 800, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 50}
                )
                if time.time() > deadline - 3.0:
                    break
                pts = res.x.reshape(8, 2)
                # Check distinct
                dd = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
                np.fill_diagonal(dd, np.inf)
                if dd.min() < 1e-7:
                    continue
                sc = _true_score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
            except Exception:
                pass
        restart += 1
    
    return [[float(best_pts[i, 0]), float(best_pts[i, 1])] for i in range(8)]
# EVOLVE-BLOCK-END
