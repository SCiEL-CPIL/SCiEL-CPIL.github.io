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
    time_limit = 620  # safety margin before 630s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x, s=0.05):
        """Smooth surrogate for maximizing (d_min/d_max)^2."""
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10
        
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        dists = pdist(p, metric='euclidean')
        if np.any(dists < 0):
            return 1e10
            
        # Soft min and soft max using log-sum-exp
        # We want to maximize (d_min / d_max)^2, so minimize (d_max / d_min)^2
        
        # Soft min: -log(sum(exp(-d_i/s))/n)
        neg_d_over_s = -dists / s
        max_neg = np.max(neg_d_over_s)
        soft_min = -s * (max_neg + np.log(np.sum(np.exp(neg_d_over_s - max_neg)) / len(dists)))
        
        # Soft max: log(sum(exp(d_i/s))/n)
        d_over_s = dists / s
        max_pos = np.max(d_over_s)
        soft_max = s * (max_pos + np.log(np.sum(np.exp(d_over_s - max_pos)) / len(dists)))
        
        if soft_min <= 1e-10 or soft_max <= 1e-10:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    # Initial configuration: icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Base initial points: icosahedron + 2 poles
    base_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    
    best_points = base_points.copy()
    best_score = get_score(base_points)
    
    # Generate multiple initializations
    initializations = [base_points]
    
    # Random initializations on sphere
    np.random.seed(42)
    for seed in range(20):
        np.random.seed(seed)
        rand_points = np.random.randn(n, d)
        norms = np.linalg.norm(rand_points, axis=1)
        rand_points = rand_points / norms[:, np.newaxis]
        initializations.append(rand_points)
        
        # Also try with some points slightly off the sphere
        if seed % 2 == 0:
            rand_points2 = rand_points.copy()
            rand_points2[0, :] *= 1.2
            rand_points2[1, :] *= 0.8
            initializations.append(rand_points2)
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
            
        try:
            # Try different temperature parameters for the soft min/max
            for s_val in [0.1, 0.05, 0.02]:
                elapsed = time.time() - start_time
                if elapsed > time_limit - 10:
                    break
                    
                # Use L-BFGS-B
                res = minimize(objective, init_pts.ravel(), method='L-BFGS-B', 
                              args=(s_val,),
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
                
                if res.fun < 1e10:
                    opt_points = res.x.reshape((n, d))
                    opt_score = get_score(opt_points)
                    
                    if opt_score > best_score:
                        best_points = opt_points.copy()
                        best_score = opt_score
                        
                        # Polish pass with tighter tolerance
                        elapsed = time.time() - start_time
                        if elapsed < time_limit - 30:
                            res2 = minimize(objective, opt_points.ravel(), method='L-BFGS-B', 
                                          args=(s_val,),
                                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000})
                            if res2.fun < 1e10:
                                polish_points = res2.x.reshape((n, d))
                                polish_score = get_score(polish_points)
                                if polish_score > best_score:
                                    best_points = polish_points.copy()
                                    best_score = polish_score
                                    
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = base_points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
