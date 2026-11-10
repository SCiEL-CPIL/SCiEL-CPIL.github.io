import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
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
    time_limit = 585  # Safety margin before 600s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if np.any(dists == 0):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(p):
        """Center and normalize scale for stability."""
        p = p - np.mean(p, axis=0)
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-15:
            return p
        return p * np.sqrt(n / norm_sq)

    # Smooth surrogate for maximizing (d_min/d_max)^2
    # Equivalent to minimizing log(d_max) - log(d_min)
    def objective(x, beta):
        """
        Minimize soft_max(log(d)) - soft_min(log(d))
        using logsumexp for numerical stability.
        beta is the inverse temperature (higher = sharper).
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10
        
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        # Center the points to remove translational invariance
        p = p - np.mean(p, axis=0)
        
        # Normalize scale to remove scaling invariance
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-12:
            return 1e10 + 1.0 / norm_sq
        
        # Fix scale: sum(p^2) = n
        p = p * np.sqrt(n / norm_sq)

        dists = pdist(p, metric='euclidean')
        
        min_dist = np.min(dists)
        if min_dist < 1e-12:
            return 1e10 + 1.0 / min_dist
            
        log_dists = np.log(dists)
        
        # Soft max/min of log_dists
        bm = beta * log_dists
        bmin = -beta * log_dists
        
        soft_max_val = (1.0 / beta) * logsumexp(bm)
        soft_min_val = -(1.0 / beta) * logsumexp(bmin)
        
        return soft_max_val - soft_min_val

    # Initial configurations
    initializations = []
    
    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))
    
    # 2. Icosahedron + 2 points slightly off poles
    offset = 0.1
    poles_off = np.array([[offset, 0, np.sqrt(1-offset**2)], [0, offset, -np.sqrt(1-offset**2)]])
    initializations.append(np.vstack([ico_verts, poles_off]))

    # 3. Fibonacci sphere (golden spiral)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        radius = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        fib_pts[i] = [x, y, z]
    initializations.append(fib_pts)
    
    # 4. More random initializations for diversity
    for seed in [42, 123, 456, 789, 101]:
        np.random.seed(seed)
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Optimization loop with adaptive beta schedule
    # Beta acts as 1/s in the previous version. Higher beta = sharper min/max approximation.
    beta_schedule = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            current_pts = init_pts.copy()
            for beta_val in beta_schedule:
                if time.time() - start_time > time_limit - 30:
                    break
                    
                res = minimize(objective, current_pts.ravel(), method='L-BFGS-B', 
                               args=(beta_val,),
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        opt_points = normalize_pts(opt_points)
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        current_pts = opt_points.copy()
                        
        except Exception:
            pass
    
    # Final high-precision polish with very large beta
    if time.time() - start_time < time_limit - 40:
        try:
            for beta_val in [2000.0, 5000.0, 10000.0]:
                if time.time() - start_time > time_limit - 20:
                    break
                res = minimize(objective, best_points.ravel(), method='L-BFGS-B', 
                               args=(beta_val,),
                               options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        opt_points = normalize_pts(opt_points)
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
        except Exception:
            pass

    # Nelder-Mead polish on the exact nonsmooth metric
    # This helps escape local minima of the smooth surrogate and directly optimize the true objective
    if time.time() - start_time < time_limit - 15:
        try:
            def exact_neg_score(x):
                elapsed = time.time() - start_time
                if elapsed > time_limit - 5:
                    return 0.0
                p = x.reshape((n, d))
                if not np.all(np.isfinite(p)):
                    return 0.0
                p = normalize_pts(p)
                return -get_score(p)
            
            res_nm = minimize(exact_neg_score, best_points.ravel(), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            # Note: res_nm.fun is the negative score, so we just check if it's valid
            if res_nm.x is not None:
                nm_points = res_nm.x.reshape((n, d))
                if np.all(np.isfinite(nm_points)):
                    nm_points = normalize_pts(nm_points)
                    nm_score = get_score(nm_points)
                    if nm_score > best_score:
                        best_score = nm_score
                        best_points = nm_points.copy()
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
        
    # Ensure all points are finite and return
    best_points = np.asarray(best_points, dtype=np.float64)
    return best_points

# EVOLVE-BLOCK-END
