"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute min triangle area over all triples
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
    return min_area / hull_area

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    rng = np.random.RandomState(N)
    
    # Start with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Local optimization using Nelder-Mead
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    def objective_fast(x):
        """Faster objective using vectorized area computation."""
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Quick check for degenerate cases
        if np.any(np.abs(pts) > 100):
            return 1.0  # Large penalty
        
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 1.0
        except Exception:
            return 1.0
        
        # Vectorized min triangle area
        min_area = float('inf')
        # Use a sampled subset for speed, or compute all if fast enough
        # For n=14, C(14,3) = 364 triangles - feasible
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        
        return min_area / hull_area
    
    # Multiple restarts with different perturbations
    for restart in range(15):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Perturb best solution
            x0 = best_points.flatten() + rng.normal(0, 0.05, size=N * 2)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            new_points = res.x.reshape(-1, 2)
            new_score = _score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
        
        # Also try random restarts
        if restart % 5 == 2 and time.time() < deadline - 30:
            x0 = rng.uniform(-1, 1, size=N * 2)
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
                new_points = res.x.reshape(-1, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
