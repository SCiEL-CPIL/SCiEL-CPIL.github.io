"""Optimize 14 points in 3D to minimize (max_dist/min_dist)^2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3
TIME_LIMIT = 110  # safety margin before 120s limit


def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances. Returns (N, N) matrix."""
    # pts: (N, D)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq_norms = np.sum(pts * pts, axis=1)
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * pts @ pts.T
    # Numerical cleanup
    sq_dists = np.maximum(sq_dists, 0.0)
    np.fill_diagonal(sq_dists, 0.0)
    return sq_dists


def _score_from_sq_dists(sq_dists):
    """Compute (max_d / min_d)^2 from squared distance matrix."""
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_dists[mask]
    d_sq = d_sq[d_sq > 1e-12]  # ignore near-zero
    if len(d_sq) == 0:
        return 1e10
    min_sq = np.min(d_sq)
    max_sq = np.max(d_sq)
    if min_sq < 1e-12:
        return 1e10
    ratio = max_sq / min_sq
    return ratio


def _objective(params):
    """Objective: (max_d/min_d)^2. params is flat array of shape (N*D,)."""
    pts = params.reshape(N, D)
    sq_dists = _pairwise_sq_dists(pts)
    return _score_from_sq_dists(sq_dists)


def _grad(params):
    """Gradient of (max_d/min_d)^2 via finite differences (for L-BFGS-B)."""
    # Use numerical gradient for simplicity; N*D = 42, so 84 evaluations
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params[i] += eps
        f1 = _objective(params)
        params[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad


def _random_init(rng):
    """Random initialization on unit sphere scaled."""
    pts = rng.standard_normal((N, D))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms  # unit sphere
    return pts.ravel()


def _grid_init():
    """Grid-like initialization for 14 points."""
    # Use a structured approach: place points on a grid in 3D
    # 14 points: 2x2x3 + 2 extra, or other arrangement
    # Let's try a simple grid: 3x3x3 has 27, take 14
    side = 3
    grid = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                grid.append([i, j, k])
    # Take first 14
    pts = np.array(grid[:N], dtype=float)
    # Center and scale
    pts -= pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale
    return pts.ravel()


def _icosa_init():
    """Icosahedron-based initialization for 12 points, add 2 more."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    # Add 2 points: place them at poles or other positions
    # For 14, add points slightly off the icosahedron
    extra1 = np.array([0, 0, 1.5])
    extra2 = np.array([0, 0, -1.5])
    pts = np.vstack([ico, extra1, extra2])
    pts -= pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale
    return pts.ravel()


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = 1e10
    
    # Initial candidates
    inits = []
    inits.append(_grid_init())
    inits.append(_icosa_init())
    for _ in range(5):
        inits.append(_random_init(rng))
    
    # Evaluate initial scores
    for init in inits:
        s = _objective(init)
        if s < best_score:
            best_score = s
            best_params = init.copy()
    
    # Optimization loop with time budget
    deadline = start_time + TIME_LIMIT
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Choose starting point
        if attempt < len(inits):
            x0 = inits[attempt].copy()
        else:
            # Perturb best solution
            x0 = best_params + rng.standard_normal(N * D) * 0.1
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10}
            )
            s = result.fun
            if s < best_score:
                best_score = s
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    # Final local refinement with tighter tolerance
    if time.time() < deadline - 3 and best_params is not None:
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12}
            )
            s = result.fun
            if s < best_score:
                best_score = s
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Return the best solution
    if best_params is None:
        # Fallback to grid
        return _grid_init().reshape(N, D)
    
    return best_params.reshape(N, D)
