import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D space that maximize the ratio of the minimum 
    pairwise distance to the maximum pairwise distance.
    
    Returns:
        np.ndarray: Array of shape (14, 3) containing the coordinates.
    """
    start_time = time.time()
    deadline = 500.0  # 10 minutes limit, leave margin
    
    n_points = 14
    dim = 3
    
    def compute_metric(points):
        """Compute the actual metric (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2

    def objective_soft(points_flat, beta):
        """
        Objective function to minimize: negative log ratio of soft-min to soft-max.
        points_flat: 1D array of size 42
        beta: temperature parameter for soft-min/max
        """
        points = points_flat.reshape(n_points, dim)
        d = pdist(points)
        
        # Use logsumexp for numerical stability
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # soft_max =  1/beta * log(sum(exp( beta * d)))
        
        log_sum_min = logsumexp(-beta * d)
        log_sum_max = logsumexp(beta * d)
        
        soft_min = -1.0 / beta * log_sum_min
        soft_max =  1.0 / beta * log_sum_max
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        # We want to maximize log(soft_min) - log(soft_max)
        # So we minimize the negative of that
        return -(np.log(soft_min) - np.log(soft_max))

    best_score = 0.0
    best_points = None
    
    # Phase 1: Multi-start coarse optimization
    n_starts = 60
    for i in range(n_starts):
        if time.time() > deadline - 60:  # Leave 60s for Phase 2
            break
            
        # Generate random initial points
        np.random.seed(1000 + i)
        points = np.random.randn(n_points, dim)
        
        # Normalize to unit diameter
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 0:
            points = points / d_max
        else:
            continue
            
        x0 = points.flatten()
        
        # Run optimization
        # Powell is robust for small dimensional spaces
        try:
            res = minimize(
                objective_soft, 
                x0, 
                args=(20,),  # beta=20 for coarse search
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.success:
                p_opt = res.x.reshape(n_points, dim)
                score = compute_metric(p_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = p_opt
                    
        except Exception:
            continue

    # Phase 2: Polish the best candidates
    # We don't store all candidates in the simple loop above, 
    # so let's do a second pass with higher beta on the best found so far,
    # and maybe a few new starts if time permits.
    
    if best_points is not None:
        # Polish the current best
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_soft, 
                x0, 
                args=(100,),  # higher beta for sharper approximation
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.success:
                p_opt = res.x.reshape(n_points, dim)
                score = compute_metric(p_opt)
                if score > best_score:
                    best_score = score
                    best_points = p_opt
        except Exception:
            pass

    # If time allows, try a few more high-precision starts
    while time.time() < deadline - 10:
        # Try 1 new start with high beta directly
        np.random.seed(int(time.time() * 1000) % 10000)
        points = np.random.randn(n_points, dim)
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 0:
            points = points / d_max
            x0 = points.flatten()
            try:
                res = minimize(
                    objective_soft, 
                    x0, 
                    args=(50,),
                    method='Powell', 
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.success:
                    p_opt = res.x.reshape(n_points, dim)
                    score = compute_metric(p_opt)
                    if score > best_score:
                        best_score = score
                        best_points = p_opt
            except Exception:
                pass
        # Add a small sleep to avoid tight loop if optimization is very fast
        time.sleep(0.1)

    if best_points is None:
        # Fallback: return random points
        np.random.seed(0)
        best_points = np.random.randn(n_points, dim)
        
    return best_points
