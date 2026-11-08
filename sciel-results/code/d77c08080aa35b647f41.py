import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    # The limit is 120s or 150s. Let's use 110s to be safe.
    time_limit = 110.0 

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return -min_area

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), rotation=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(11, 1.0)
        # Inner 2 points: antipodal
        inner = _regular_polygon(2, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 6 inner + 1 center
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(6, 1.0)
        inner = _regular_polygon(6, r)
        center = np.array([[0.0, 0.0]])
        initializations.append(np.vstack([outer, inner, center]))
        
    # 9. 5 outer + 8 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 10. 4 outer + 9 inner
    for r in [0.2, 0.4, 0.6, 0.8]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r)
        initializations.append(np.vstack([outer, inner]))
        
    # Add random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = rng.randn(N, 2)
        # Normalize to unit area to ensure good scale
        pts = _normalize_points(pts)
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    # Optimization strategy:
    # Direct search on Hard Objective using Nelder-Mead
    # This is more robust to the non-smooth "min" function than gradient-based soft-min
    
    # Sort initializations by score to prioritize good starts
    scored_inits = []
    for init in initializations:
        score = compute_score(init)
        scored_inits.append((score, init))
    
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # We will try the top K initializations
    # Nelder-Mead is expensive per step, so we limit the number of starts
    # but give each start a reasonable number of iterations
    
    max_starts = 20
    for i, (init_score, init) in enumerate(scored_inits[:max_starts]):
        if time.time() > start_time + time_limit - 5:
            break
            
        current = init.copy().flatten()
        
        try:
            res = opt.minimize(
                _hard_objective,
                current,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
                
            # If we found a very good score, we can stop early?
            # The theoretical max for 13 points is likely around 0.05-0.08?
            # Let's just continue to ensure we find the best local optima.
            
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = scored_inits[0][1]

    return best_points
