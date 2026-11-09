# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _min_dist_on_sphere(points: np.ndarray) -> float:
    """Return the minimum pairwise distance for points assumed to be on unit sphere."""
    dists = pdist(points)
    return np.min(dists)


def _negative_min_dist(points: np.ndarray) -> float:
    """Objective to minimize (negative of min distance) for optimization."""
    return -_min_dist_on_sphere(points)


def _project_to_sphere(points: np.ndarray) -> np.ndarray:
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms


def _try_configuration(initial_points: np.ndarray, max_iter: int = 5000, tol: float = 1e-10) -> tuple:
    """Try to optimize a configuration using Nelder-Mead on the sphere."""
    n = initial_points.shape[0]
    
    # Convert to spherical coordinates for optimization
    # Use azimuth (theta) and elevation (phi) for each point
    def to_spherical(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # azimuth
        phi = np.arccos(np.clip(z / r, -1, 1))  # polar angle
        return np.column_stack([theta, phi])
    
    def from_spherical(params):
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    init_sph = to_spherical(initial_points)
    
    def objective(params):
        pts = from_spherical(params)
        return -_min_dist_on_sphere(pts)
    
    # Use Nelder-Mead
    result = minimize(objective, init_sph, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
    
    optimized_pts = from_spherical(result.x)
    score = _min_dist_on_sphere(optimized_pts)
    
    return optimized_pts, score, result.fun


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 550  # safety margin
    
    # Start with icosahedron (12 points) + 2 additional points
    ico = _icosahedron_vertices()
    
    # Try different initial positions for the 2 additional points
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Add points at ±z axis (poles)
    # But icosahedron already has points near poles, so try other positions
    
    # Generate candidate initial configurations
    candidates = []
    
    # Candidate 1: icosahedron + 2 points at (0,0,1) and (0,0,-1)
    # But these might be too close to existing vertices
    # Let's try symmetric additions
    
    # For 14 points, a good approach is to use the icosahedron and add 2 points
    # that break the symmetry in a balanced way
    
    # Try adding points at various symmetric positions
    test_positions = [
        # Opposite pairs
        (np.array([1, 0, 0]), np.array([-1, 0, 0])),
        (np.array([0, 1, 0]), np.array([0, -1, 0])),
        (np.array([0, 0, 1]), np.array([0, 0, -1])),
        # Diagonal pairs
        (np.array([1, 1, 0])/np.sqrt(2), np.array([-1, -1, 0])/np.sqrt(2)),
        (np.array([1, 0, 1])/np.sqrt(2), np.array([-1, 0, -1])/np.sqrt(2)),
        (np.array([0, 1, 1])/np.sqrt(2), np.array([0, -1, -1])/np.sqrt(2)),
        # Other symmetric pairs
        (np.array([1, 1, 1])/np.sqrt(3), np.array([-1, -1, -1])/np.sqrt(3)),
        (np.array([1, 1, -1])/np.sqrt(3), np.array([-1, -1, 1])/np.sqrt(3)),
        (np.array([1, -1, 1])/np.sqrt(3), np.array([-1, 1, -1])/np.sqrt(3)),
    ]
    
    for p1, p2 in test_positions:
        candidate = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
        # Normalize to unit sphere
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        candidate = candidate / norms
        candidates.append(candidate)
    
    # Also try random initializations
    rng = np.random.RandomState(42)
    for _ in range(20):
        rand_pts = rng.randn(14, 3)
        norms = np.linalg.norm(rand_pts, axis=1, keepdims=True)
        rand_pts = rand_pts / norms
        candidates.append(rand_pts)
    
    # Optimize each candidate
    for i, candidate in enumerate(candidates):
        if time.time() > deadline:
            break
        
        try:
            optimized_pts, score, _ = _try_configuration(candidate, max_iter=2000)
            if score > best_score:
                best_score = score
                best_points = optimized_pts
        except Exception:
            continue
    
    # If we still have time, do more random restarts
    if best_points is None:
        best_points = candidates[0]
        best_score = _min_dist_on_sphere(best_points)
    
    # Final: scale so that max distance is reasonable (metric is scale-invariant, so any scale works)
    # But we need finite coordinates. Let's keep them on unit sphere.
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        # Fallback to icosahedron + 2 points
        best_points = np.vstack([ico, np.array([[0, 0, 1]]), np.array([[0, 0, -1]])])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points


# EVOLVE-BLOCK-END
