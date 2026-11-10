"""Fifty equal circles in a circle – fresh design: ring-based analytic start + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Known best packing radius (Graham et al. 1998)
R_KNOWN = 0.1258254895304


def _score_from_centers(c):
    """Compute the common radius from a set of centers."""
    n = len(c)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to boundary
    r = np.linalg.norm(c, axis=1)
    min_bound = 1.0 - np.max(r)
    return min(min_pair, min_bound)


def _objective(x):
    """Negative of the common radius (for minimization)."""
    c = x.reshape(-1, 2)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    r = np.linalg.norm(c, axis=1)
    min_bound = 1.0 - np.max(r)
    return -min(min_pair, min_bound)


def _initial_ring_layout():
    """
    Create an initial layout based on concentric rings.
    Ring 0: center
    Ring 1: 6 points
    Ring 2: 12 points
    Ring 3: 15 points (approx)
    Ring 4: remaining
    """
    points = []
    # Center
    points.append(np.array([0.0, 0.0]))
    
    # Ring 1: 6 points at radius ~0.25
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle)]))
    
    # Ring 2: 12 points at radius ~0.5
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12
        points.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle)]))
    
    # Ring 3: 15 points at radius ~0.75
    r3 = 0.75
    for i in range(15):
        angle = 2 * np.pi * i / 15
        points.append(np.array([r3 * np.cos(angle), r3 * np.sin(angle)]))
    
    # Ring 4: remaining 16 points at radius ~0.95
    r4 = 0.95
    remaining = N - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        points.append(np.array([r4 * np.cos(angle), r4 * np.sin(angle)]))
    
    return np.array(points)


def _initial_hexagonal():
    """Hexagonal lattice clipped to unit disk, take N closest to center."""
    spacing = 0.26
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                points.append([x, y])
    points = np.array(points)
    r = np.linalg.norm(points, axis=1)
    idx = np.argsort(r)
    return points[idx[:N]]


def _initial_random():
    """Random points in unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.95, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _optimize_from_start(x0, maxiter=2000, timeout=100):
    """Run L-BFGS-B from a starting point. Returns (best_x, best_score)."""
    start_time = time.time()
    
    x0 = np.array(x0, dtype=float).flatten()
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_x = x0.copy()
    best_score = -_objective(x0)
    
    def _callback(xk):
        nonlocal best_x, best_score
        s = -_objective(xk)
        if s > best_score:
            best_score = s
            best_x = xk.copy()
        return (time.time() - start_time) > timeout
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10},
            callback=_callback
        )
        s = -_objective(result.x)
        if s > best_score:
            best_score = s
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(-1, 2), best_score


def construct():
    """Construct centers of N equal circles inside the unit circle."""
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_centers = None
    best_score = -1e9
    
    # Try multiple initializations
    initializers = [
        ("ring", _initial_ring_layout),
        ("hex", _initial_hexagonal),
    ]
    
    # Add a few random starts
    rng = np.random.RandomState(42)
    for seed in range(3):
        initializers.append((f"rand{seed}", lambda s=seed: _initial_random()))
    
    for name, init_fn in initializers:
        if time.time() > deadline - 20:
            break
        
        try:
            x0 = init_fn()
            if x0 is None or len(x0) != N:
                continue
            
            centers, score = _optimize_from_start(x0, maxiter=1500, timeout=100)
            
            if score > best_score:
                best_score = score
                best_centers = centers
        except Exception:
            continue
    
    # If we still have time, do a few more random restarts
    while time.time() < deadline - 15:
        try:
            x0 = _initial_random()
            centers, score = _optimize_from_start(x0, maxiter=800, timeout=100)
            if score > best_score:
                best_score = score
                best_centers = centers
        except Exception:
            continue
    
    if best_centers is None:
        # Fallback: hexagonal
        best_centers = _initial_hexagonal()
    
    # Final validation
    if not np.all(np.isfinite(best_centers)):
        best_centers = _initial_hexagonal()
    
    return best_centers
