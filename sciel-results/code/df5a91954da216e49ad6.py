"""Optimized placement of 14 points in a convex region to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
DEADLINE = time.time() + 140.0  # Safety margin before 150s limit


def _convex_hull_area(points):
    """Compute area of convex hull of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _flatten(points):
    return points.reshape(-1)


def _unflatten(flat):
    return flat.reshape(N, 2)


def _initial_config():
    """Generate a reasonable initial configuration using two rings."""
    # Outer ring: 7 points roughly evenly spaced on a circle
    # Inner ring: 7 points roughly evenly spaced on a smaller circle, rotated
    n_outer = 7
    n_inner = 7
    points = []
    # Outer ring
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        points.append([np.cos(angle), np.sin(angle)])
    # Inner ring, rotated by half a step
    r_inner = 0.5
    for i in range(n_inner):
        angle = 2 * np.pi * (i + 0.5) / n_inner
        points.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    return np.array(points, dtype=np.float64)


def _fourier_initial():
    """Initial config with slight Fourier distortion for asymmetry."""
    base = _initial_config()
    # Add small perturbations
    rng = np.random.RandomState(42)
    base += 0.05 * rng.randn(N, 2)
    return base


def _local_optimize(points, max_iter=200, tol=1e-10):
    """Local optimization using L-BFGS-B."""
    flat = _flatten(points)
    
    def obj(x):
        if time.time() > DEADLINE:
            return 1e10
        return _neg_score(_unflatten(x))
    
    try:
        res = minimize(obj, flat, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol})
        if res.success or res.fun < _neg_score(points):
            return _unflatten(res.x)
    except Exception:
        pass
    return points


def _coordinate_descent(points, max_rounds=50):
    """Simple coordinate descent: move each point slightly to improve score."""
    best = points.copy()
    best_score = _score(best)
    
    for rnd in range(max_rounds):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            if time.time() > DEADLINE:
                break
            # Try small perturbations
            for dx in [-0.02, 0.02]:
                for dy in [-0.02, 0.02]:
                    candidate = best.copy()
                    candidate[i, 0] += dx
                    candidate[i, 1] += dy
                    s = _score(candidate)
                    if s > best_score + 1e-12:
                        best = candidate
                        best_score = s
                        improved = True
        if not improved:
            break
    
    return best


def construct():
    """Construct 14 points maximizing min triangle area / hull area."""
    start_time = time.time()
    
    # Start with a good initial configuration
    best_points = _initial_config()
    best_score = _score(best_points)
    
    # Local optimization from initial config
    optimized = _local_optimize(best_points, max_iter=300)
    opt_score = _score(optimized)
    if opt_score > best_score:
        best_points = optimized
        best_score = opt_score
    
    # Try coordinate descent polish
    polished = _coordinate_descent(best_points, max_rounds=30)
    pol_score = _score(polished)
    if pol_score > best_score:
        best_points = polished
        best_score = pol_score
    
    # Try differential evolution with limited budget
    if time.time() < DEADLINE - 30:
        # Use a bounded box
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        def de_obj(x):
            if time.time() > DEADLINE:
                return 1e10
            return _neg_score(_unflatten(x))
        
        try:
            de_res = differential_evolution(de_obj, bounds, 
                                            maxiter=50, popsize=15,
                                            seed=42, tol=1e-8,
                                            init='latinhypercube',
                                            polish=False)
            de_points = _unflatten(de_res.x)
            de_score = _score(de_points)
            if de_score > best_score:
                best_points = de_points
                best_score = de_score
                # Polish the DE result
                if time.time() < DEADLINE - 10:
                    polished_de = _local_optimize(de_points, max_iter=200)
                    pol_de_score = _score(polished_de)
                    if pol_de_score > best_score:
                        best_points = polished_de
                        best_score = pol_de_score
        except Exception:
            pass
    
    # Final local polish
    if time.time() < DEADLINE - 5:
        final = _local_optimize(best_points, max_iter=100)
        final_score = _score(final)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    # Ensure points are in a reasonable range
    # Normalize to unit area if needed (the score already normalizes by hull area)
    
    return best_points
