# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds

    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0
        return (d_min / d_max) ** 2

    # Initial configuration: 12 vertices of a regular icosahedron + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize icosahedron to unit sphere
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Initial 14 points
    points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    
    def objective(x, s=0.01):
        """Smooth surrogate for maximizing (d_min/d_max)^2 using log-sum-exp."""
        if time.time() - start_time > time_limit - 10:
            return 1e10
        
        p = x.reshape((n, d))
        dists = pdist(p, metric='euclidean')
        
        # Soft min: -log(sum(exp(-d_i/s))/n) approximates min(d_i)
        # Soft max: log(sum(exp(d_i/s))/n) approximates max(d_i)
        # We want to maximize (soft_min / soft_max)^2, so minimize (soft_max / soft_min)^2
        
        # For numerical stability:
        neg_d_over_s = -dists / s
        max_neg = np.max(neg_d_over_s)
        soft_min = -s * (max_neg + np.log(np.sum(np.exp(neg_d_over_s - max_neg)) / n))
        
        d_over_s = dists / s
        max_pos = np.max(d_over_s)
        soft_max = s * (max_pos + np.log(np.sum(np.exp(d_over_s - max_pos)) / n))
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
        
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def objective_tight(x, s=0.001):
        """Tighter soft surrogate for final polishing."""
        if time.time() - start_time > time_limit - 10:
            return 1e10
        
        p = x.reshape((n, d))
        dists = pdist(p, metric='euclidean')
        
        neg_d_over_s = -dists / s
        max_neg = np.max(neg_d_over_s)
        soft_min = -s * (max_neg + np.log(np.sum(np.exp(neg_d_over_s - max_neg)) / n))
        
        d_over_s = dists / s
        max_pos = np.max(d_over_s)
        soft_max = s * (max_pos + np.log(np.sum(np.exp(d_over_s - max_pos)) / n))
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
        
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    # Optimize
    best_points = points.copy()
    best_score = get_score(points)
    
    # Try multiple initial conditions
    initializations = [
        points,
        np.vstack([ico_verts_norm, np.array([[0, 0, 0.5], [0, 0, -0.5]])]),
        np.vstack([ico_verts_norm, np.array([[0, 0, 2.0], [0, 0, -2.0]])]),
        np.vstack([ico_verts_norm, np.array([[0, 1, 0], [0, -1, 0]])]),
        np.vstack([ico_verts_norm, np.array([[1, 0, 0], [-1, 0, 0]])]),
    ]
    
    # Add random initializations with different seeds
    np.random.seed(42)
    for seed in [42, 123, 456, 789, 1011, 2020]:
        np.random.seed(seed)
        rand_points = np.random.randn(n, d)
        rand_points = rand_points / np.linalg.norm(rand_points, axis=1)[:, np.newaxis]
        initializations.append(rand_points)
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 30:
            break
        
        try:
            # First pass with moderate softness
            res = minimize(objective, init_pts.ravel(), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e10:
                opt_points = res.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points.copy()
                    best_score = opt_score
                    
                    # Second pass with tighter softness
                    if time.time() - start_time < time_limit - 60:
                        res2 = minimize(lambda x: objective(x, s=0.005), opt_points.ravel(), 
                                       method='L-BFGS-B',
                                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                        if res2.fun < 1e10:
                            polish_points = res2.x.reshape((n, d))
                            polish_score = get_score(polish_points)
                            if polish_score > best_score:
                                best_points = polish_points.copy()
                                best_score = polish_score
                                
                                # Third pass with very tight softness
                                if time.time() - start_time < time_limit - 30:
                                    res3 = minimize(lambda x: objective_tight(x, s=0.001), 
                                                   polish_points.ravel(),
                                                   method='L-BFGS-B',
                                                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
                                    if res3.fun < 1e10:
                                        final_points = res3.x.reshape((n, d))
                                        final_score = get_score(final_points)
                                        if final_score > best_score:
                                            best_points = final_points.copy()
                                            best_score = final_score
        except Exception:
            pass

    # Final exact refinement: use actual min/max to guide a very local search
    if time.time() - start_time < time_limit - 20:
        try:
            def exact_objective(x):
                if time.time() - start_time > time_limit - 10:
                    return 1e10
                p = x.reshape((n, d))
                dists = pdist(p, metric='euclidean')
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max == 0:
                    return 1e10
                return (d_max / d_min) ** 2
            
            res_final = minimize(exact_objective, best_points.ravel(), method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_final.fun < 1e10:
                final_pts = res_final.x.reshape((n, d))
                final_sc = get_score(final_pts)
                if final_sc > best_score:
                    best_points = final_pts.copy()
                    best_score = final_sc
        except Exception:
            pass

    # Final check for validity
    if not np.all(np.isfinite(best_points)):
        best_points = points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
