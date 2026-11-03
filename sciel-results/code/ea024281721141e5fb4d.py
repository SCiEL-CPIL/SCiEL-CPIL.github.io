"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14
HALF_N = N // 2  # 7 points in one half; the other half is -points (central symmetry)

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            pi = pts[i]
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = pts.shape[0]
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a

def _full_score(half_pts):
    """Given 7 points (one half), construct full 14-point set with central symmetry."""
    full = np.vstack([half_pts, -half_pts])
    return _score(full)

def _objective(half_flat):
    """Objective for optimization: negative score (we minimize)."""
    half_pts = half_flat.reshape(-1, 2)
    s = _full_score(half_pts)
    return -s

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    best_score = 0.0
    best_half = None
    
    # Strategy: Differential Evolution on 14D space (7 points x 2 coords)
    # Use central symmetry: 14 points = 7 + their negations
    
    from scipy.optimize import differential_evolution
    
    # Initial bounds: points in [-2, 2] to allow flexibility
    bounds = [(-2.0, 2.0)] * (2 * HALF_N)
    
    def _time_check():
        return time.time() < deadline
    
    # Run DE with multiple seeds for diversity
    seeds = [42, 123, 777, 2024, 999]
    
    for seed in seeds:
        if not _time_check():
            break
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=seed,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            
            if not _time_check():
                break
            
            half_pts = result.x.reshape(-1, 2)
            s = _full_score(half_pts)
            
            if s > best_score:
                best_score = s
                best_half = half_pts.copy()
                
        except Exception:
            continue
    
    # If DE didn't find anything good, try a grid-based initial guess
    if best_half is None:
        # Regular heptagon on a circle
        angles = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False)
        radius = 1.0
        best_half = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        best_score = _full_score(best_half)
    
    # Local refinement with Nelder-Mead on the best solution
    if best_half is not None and _time_check():
        from scipy.optimize import minimize
        
        try:
            res = minimize(
                _objective,
                best_half.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.x is not None:
                half_pts = res.x.reshape(-1, 2)
                s = _full_score(half_pts)
                if s > best_score:
                    best_score = s
                    best_half = half_pts.copy()
        except Exception:
            pass
    
    # Final check: if we still have time, try a few more random restarts with local opt
    if _time_check() and best_half is not None:
        rng = np.random.default_rng(12345)
        for _ in range(5):
            if not _time_check():
                break
            try:
                # Random perturbation around best
                perturbed = best_half + rng.normal(0, 0.1, size=best_half.shape)
                res = minimize(
                    _objective,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.x is not None:
                    half_pts = res.x.reshape(-1, 2)
                    s = _full_score(half_pts)
                    if s > best_score:
                        best_score = s
                        best_half = half_pts.copy()
            except Exception:
                continue
    
    # Construct final 14-point set
    full = np.vstack([best_half, -best_half])
    
    # Normalize: the score is min_triangle_area / hull_area, which is scale-invariant
    # So we can return the points as-is; the evaluator will compute the ratio.
    # But let's normalize to unit area for cleanliness (doesn't change score)
    hull_a = _hull_area(full)
    if hull_a > 0:
        full = full / np.sqrt(hull_a)  # Scale so hull area = 1
    
    return full
