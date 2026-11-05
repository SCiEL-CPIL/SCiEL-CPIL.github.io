"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(points):
    """Compute the scale-invariant score: product of all ordered-pair distances / diameter^56."""
    n = len(points)
    pts = np.array(points)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out diagonal (zero distances)
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    # Product of all distances / diameter^(n*(n-1))
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)

def _neg_log_score(points_flat):
    """Negative log of the scale-invariant score for optimization."""
    n = 8
    pts = points_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return -log_score

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Start with regular octagon
    angles = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _score(octagon)
    
    # Try optimization from the regular octagon
    x0 = octagon.flatten()
    try:
        result = minimize(_neg_log_score, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-12})
        if result.success or result.fun < _neg_log_score(x0):
            candidate = result.x.reshape(8, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best found so far, scaled to diameter 2
        pts = np.array(best_pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        mask = ~np.eye(8, dtype=bool)
        diameter = np.max(dists[mask])
        if diameter > 1e-12:
            pts = pts * (2.0 / diameter)
        return [[float(p[0]), float(p[1])] for p in pts]
    
    # Try multiple random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb the best configuration
        noise = rng.normal(0, 0.3, size=(8, 2))
        x0 = (best_pts + noise).flatten()
        try:
            result = minimize(_neg_log_score, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-12})
            if result.success:
                candidate = result.x.reshape(8, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    # Scale to diameter 2
    pts = np.array(best_pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(8, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)
    
    return [[float(p[0]), float(p[1])] for p in pts]
# EVOLVE-BLOCK-END
