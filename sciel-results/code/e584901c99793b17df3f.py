# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _compute_score(points):
    """Compute A_min / A_hull ratio."""
    n = len(points)
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                area = 0.5 * abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if area < min_area:
                    min_area = area
    
    # Compute hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 0 or min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    Uses simulated annealing starting from a regular 13-gon.
    """
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    n = 13
    # Start with regular 13-gon, normalized to unit hull area
    points = _regular_polygon(n, radius=1.0)
    
    # Normalize to unit hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
    except Exception:
        pass
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Simulated annealing parameters
    T = 0.05  # Initial temperature
    cooling = 0.999
    min_T = 1e-8
    
    rng = np.random.default_rng(seed=12345)
    
    while time.time() < deadline and T > min_T:
        # Generate a perturbation
        idx = rng.integers(0, n)
        # Random step size that decreases with temperature
        step_size = T * 0.5
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_points = points.copy()
        new_points[idx] += [dx, dy]
        
        # Ensure all points are finite
        if not np.all(np.isfinite(new_points)):
            continue
        
        new_score = _compute_score(new_points)
        old_score = _compute_score(points)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score >= old_score:
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            delta = new_score - old_score
            if delta < 0:
                acceptance_prob = np.exp(delta / T)
                if rng.random() < acceptance_prob:
                    points = new_points
        
        T *= cooling
    
    # Final local refinement: try small perturbations on each point
    if time.time() < deadline - 30:
        for _ in range(50):  # 50 refinement passes
            if time.time() >= deadline - 30:
                break
            for idx in range(n):
                for trial in range(10):
                    step = 0.01 * (1 + 0.5 * trial)
                    angles = rng.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angles)
                    dy = step * np.sin(angles)
                    
                    new_points = points.copy()
                    new_points[idx] += [dx, dy]
                    
                    if not np.all(np.isfinite(new_points)):
                        continue
                    
                    new_score = _compute_score(new_points)
                    if new_score > _compute_score(points):
                        points = new_points
                        if new_score > best_score:
                            best_score = new_score
                            best_points = points.copy()
    
    return best_points


# EVOLVE-BLOCK-END
