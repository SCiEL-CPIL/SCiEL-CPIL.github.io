# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute indices for all triangles."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_score(points, tri_idx):
    """
    Compute Heilbronn score.
    Returns score.
    """
    n = len(points)
    if n < 3:
        return 0.0
        
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    score = A_min / A_hull
    return score

def _softmin_objective(points, tri_idx, beta):
    """
    Smooth surrogate for optimization: -log(softmin(areas)) - log(hull_area)
    """
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
        if A_hull <= 1e-12:
            return 1e10
    except Exception:
        return 1e10

    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    if np.any(areas < 1e-12):
         return 1e10

    # Soft min calculation
    z = -beta * areas
    z_max = np.max(z)
    exp_sum = np.sum(np.exp(z - z_max))
    log_sum_exp = z_max + np.log(exp_sum)
    soft_min_val = -log_sum_exp / beta
    
    if soft_min_val <= 1e-12:
        return 1e10
        
    return -np.log(soft_min_val) - np.log(A_hull)

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    tri_idx = _precompute_indices(n)
    best_score = 0.0
    best_points = None
    
    # Strategy:
    # 1. Fix a cyclic order of indices (0..12). 
    # 2. Parametrize points as (r_i * cos(theta_i), r_i * sin(theta_i)) where theta_i is fixed by index 
    #    and r_i is the variable. 
    #    Wait, fixing theta_i to uniform angles restricts the shape too much (concentric circles).
    #    Better: Fix the *cyclic order* of points on the convex hull. 
    #    However, not all points need to be on the hull. 
    #    
    #    Alternative Structural Approach:
    #    The optimal Heilbronn configuration for small N often has high symmetry or specific structures.
    #    For N=13, let's try optimizing the coordinates directly but using Differential Evolution 
    #    which is robust to local minima, combined with a strong soft-min objective.
    #    
    #    Why DE? The current SA+Nelder-Mead is stochastic and can get stuck. DE is a population-based 
    #    global optimizer that is often better for multimodal landscapes.
    #    To speed it up, we use a soft-min objective which is smooth.
    
    # Define the objective function for DE
    def objective(x):
        pts = x.reshape((n, 2))
        # Add a small penalty if points are too close to each other to avoid degenerate triangles
        # But the soft-min handles this.
        return _softmin_objective(pts, tri_idx, beta=50.0)

    # Bounds for coordinates: roughly [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    # Run Differential Evolution
    # maxiter is limited by time. popsize=10, strategy='best1bin'
    # We will monitor time.
    
    # Initial population: mix of random and some structured inits
    # DE generates its own population, but we can seed it? 
    # scipy DE doesn't easily accept a custom initial population in older versions, 
    # but we can just let it run.
    
    # To make it faster, we reduce popsize and maxiter, and rely on the smoothness of soft-min.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4,
            seed=42,
            polish=False,  # We will do our own polish
            init='sobol' # Sobol sequence for better coverage
        )
        
        if res.success or res.fun < 1e9:
            pts = res.x.reshape((n, 2))
            score = _compute_score(pts, tri_idx)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    except Exception:
        pass

    # If DE didn't find anything good, try a few random starts with Nelder-Mead on soft-min
    if best_score < 0.01:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 60:
                break
            init_pts = rng.uniform(-1, 1, (n, 2))
            
            def soft_obj(x):
                p = x.reshape((n, 2))
                return _softmin_objective(p, tri_idx, beta=50.0)
            
            try:
                res_nm = minimize(soft_obj, init_pts.flatten(), method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                pts = res_nm.x.reshape((n, 2))
                score = _compute_score(pts, tri_idx)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            except Exception:
                pass

    # Final Polish:
    # Now that we have a good candidate, we do a high-precision optimization on the RAW score
    # using a very small step size Nelder-Mead, or better, a local search on the hard min.
    # Since the hard min is non-differentiable, we use a "jittered" local search or just
    # a very tight Nelder-Mead on the soft-min with high beta.
    
    if best_points is not None and time.time() < deadline - 30:
        # Use high beta for soft min to approximate hard min closely
        def final_obj(x):
            p = x.reshape((n, 2))
            return _softmin_objective(p, tri_idx, beta=500.0)
        
        try:
            # Multiple restarts of Nelder-Mead from the best point found
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                # Add tiny noise to escape potential flat spots
                noise = np.random.default_rng(0).normal(0, 0.01, (n, 2))
                start_pts = best_points + noise
                
                res_final = minimize(final_obj, start_pts.flatten(), method='Nelder-Mead', 
                                     options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                pts = res_final.x.reshape((n, 2))
                score = _compute_score(pts, tri_idx)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
