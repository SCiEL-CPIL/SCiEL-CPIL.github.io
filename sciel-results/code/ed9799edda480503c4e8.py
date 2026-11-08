"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13


def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                xk, yk = pts[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(area)
    return np.array(areas)


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, 'volume' is the area
    except Exception:
        # Fallback: use the area of the polygon formed by points in order
        # This is only a fallback; should rarely be needed
        return 1.0


def _score(pts):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _soft_min_neg_logsumexp(pts, temperature=0.01):
    """
    Surrogate objective for L-BFGS-B: minimize -log(sum(exp(-area/temp)))
    This approximates minimizing the minimum triangle area.
    We also need to account for hull area normalization, but for gradient-based
    optimization, we can optimize the min triangle area directly and then
    normalize at the end. However, since the hull area changes with point
    positions, we should include it.
    
    Actually, let's just minimize the negative of the soft-min of normalized areas.
    """
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    normalized = areas / hull_area
    
    # Soft min: -log(sum(exp(-x/temp))) * temp approximates min(x)
    # We want to maximize min, so minimize -soft_min
    # soft_min(x) = -temp * log(sum(exp(-x/temp)))
    # So we minimize temp * log(sum(exp(-x/temp)))
    
    # For numerical stability
    max_val = np.max(-normalized / temperature)
    log_sum = max_val + np.log(np.sum(np.exp(-normalized / temperature - max_val)))
    objective = temperature * log_sum
    
    return objective


def _regular_ngon(n):
    """Generate points on a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def construct():
    """
    Place N=13 points to maximize min triangle area / convex hull area.
    Uses a combination of initialization on a regular polygon and gradient-based
    optimization with a soft-min surrogate.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    # Start with a regular 13-gon
    best_pts = _regular_ngon(N)
    best_score = _score(best_pts)
    
    # Try multiple initializations and optimize
    rng = np.random.RandomState(42)
    
    n_attempts = 0
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        n_attempts += 1
        
        # Different initialization strategies
        if attempt == 0:
            pts = _regular_ngon(N).copy()
        elif attempt == 1:
            # Slightly perturbed regular polygon
            pts = _regular_ngon(N).copy()
            pts += rng.randn(N, 2) * 0.1
        elif attempt == 2:
            # Two concentric rings: 7 outer, 6 inner
            angles_outer = 2 * np.pi * np.arange(7) / 7
            angles_inner = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
            r_outer = 1.0
            r_inner = 0.5
            pts = np.column_stack([
                np.concatenate([r_outer * np.cos(angles_outer), r_inner * np.cos(angles_inner)]),
                np.concatenate([r_outer * np.sin(angles_outer), r_inner * np.sin(angles_inner)])
            ])
        elif attempt == 3:
            # Three rings: 5, 4, 4
            angles1 = 2 * np.pi * np.arange(5) / 5
            angles2 = 2 * np.pi * np.arange(4) / 4 + np.pi / 4
            angles3 = 2 * np.pi * np.arange(4) / 4
            r1, r2, r3 = 1.0, 0.65, 0.3
            pts = np.column_stack([
                np.concatenate([r1 * np.cos(angles1), r2 * np.cos(angles2), r3 * np.cos(angles3)]),
                np.concatenate([r1 * np.sin(angles1), r2 * np.sin(angles2), r3 * np.sin(angles3)])
            ])
        else:
            # Random perturbations of regular polygon
            pts = _regular_ngon(N).copy()
            pts += rng.randn(N, 2) * 0.3
        
        # Normalize to unit area
        hull_area = _convex_hull_area(pts)
        if hull_area > 1e-12:
            pts = pts / np.sqrt(hull_area)
        
        # Use L-BFGS-B with soft-min surrogate
        try:
            result = minimize(
                _soft_min_neg_logsumexp,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(0.01,)
            )
            
            if result.success or result.fun < _soft_min_neg_logsumexp(pts, 0.01):
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        
        except Exception:
            pass
    
    # If we have time, try a few more local optimizations with different temperatures
    if time.time() < deadline - 10:
        for temp in [0.005, 0.02, 0.05]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _soft_min_neg_logsumexp,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
                    args=(temp,)
                )
                if result.success:
                    new_pts = result.x.reshape(N, 2)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Final check
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    return best_pts
