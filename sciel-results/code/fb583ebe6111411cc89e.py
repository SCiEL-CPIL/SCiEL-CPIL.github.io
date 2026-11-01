"""Packing 50 equal circles in a unit circle - multi-restart force-based optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_pair, r_cont)


def _objective(x):
    """Negative packing radius to minimize."""
    pts = x.reshape(N, 2)
    # Boundary violation penalty
    dists = np.hypot(pts[:, 0], pts[:, 1])
    boundary_viol = np.maximum(0, dists - 1.0)
    # Pairwise distance
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r = min(min_pair / 2.0, 1.0 - np.max(dists))
    penalty = 100.0 * np.sum(boundary_viol ** 2)
    return -r + penalty


def _hex_lattice_init(seed=0):
    """Generate a hexagonal lattice initial configuration."""
    rng = np.random.RandomState(seed)
    # Create a hexagonal grid
    spacing = 0.2
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 0.9:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # Fallback to random
        return rng.uniform(-0.8, 0.8, size=(N, 2))
    # Select N points
    idx = rng.choice(len(points), size=N, replace=False)
    return points[idx]


def _random_init(seed=0):
    """Random initial configuration inside unit circle."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, size=N)
    radii = rng.uniform(0, 0.9, size=N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def construct():
    """Construct optimal packing of 50 equal circles in unit circle."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations and optimizations
    for trial in range(20):
        if time.time() > _DEADLINE - 5.0:
            break
        
        if trial == 0:
            x0 = _hex_lattice_init(seed=trial)
        elif trial < 5:
            x0 = _random_init(seed=trial)
        else:
            # Perturb the best solution found so far
            if best_pts is not None:
                rng = np.random.RandomState(trial)
                x0 = best_pts + rng.normal(0, 0.05, size=(N, 2))
                # Clip to unit circle
                dists = np.hypot(x0[:, 0], x0[:, 1])
                mask = dists > 0.95
                x0[mask] *= 0.95 / dists[mask]
            else:
                x0 = _random_init(seed=trial)
        
        # Flatten for scipy
        x0_flat = x0.flatten()
        
        # Use L-BFGS-B with bounds
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(_objective, x0_flat, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            pts = result.x.reshape(N, 2)
            sc = _score(pts)
            
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        except Exception:
            continue
    
    # If no valid solution found, return a simple valid configuration
    if best_pts is None or _score(best_pts) <= 0:
        # Fallback: hexagonal lattice
        spacing = 0.18
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            best_pts = points[:N]
        else:
            # Last resort: random
            rng = np.random.RandomState(0)
            angles = rng.uniform(0, 2 * np.pi, size=N)
            radii = rng.uniform(0, 0.8, size=N)
            best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_pts
