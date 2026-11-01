"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _objective(points_flat):
    """Objective function for optimization (negative score to maximize)."""
    points = points_flat.reshape(-1, 2)
    return -_compute_score(points)

def construct():
    # Hexagonal lattice cut to the unit disk: valid but improvable. N is set by the task.
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Local optimization to improve the packing
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    # Try a few optimization steps
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Use gradient-free optimization with a time limit
    try:
        result = minimize(
            _objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < -best_score:
            candidate = result.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
    except Exception:
        pass
    
    # Check if we're still within time budget
    if time.time() < deadline - 30.0:
        # Try a second round with smaller steps
        try:
            result2 = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'initial_simplex': best_points.flatten() + 1e-6 * np.random.randn(2*N)}
            )
            if result2.fun < -best_score:
                candidate = result2.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
