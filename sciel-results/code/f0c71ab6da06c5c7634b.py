# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def exact_score(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_objective(x):
        return -exact_score(x)

    def soft_min(dists, t=0.1):
        """Differentiable approximation of the minimum distance."""
        return -np.log(np.sum(np.exp(-dists / t)) / len(dists)) * t

    def smooth_objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        d_min_norm = soft_min(dists_norm)
        return -d_min_norm

    bounds = [(-1, 1)] * (n * d)
    
    best_x = None
    best_score = 0.0

    # Phase 1: Global Search with Differential Evolution
    try:
        if time.time() < deadline - 20:
            res_de = differential_evolution(
                exact_objective, 
                bounds, 
                maxiter=200, 
                popsize=15, 
                seed=42, 
                tol=1e-5,
                init='latinhypercube',
                polish=False
            )
            
            if time.time() < deadline - 10:
                candidate_x = res_de.x
                candidate_score = -res_de.fun
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = candidate_x
    except Exception:
        pass

    # Phase 2: Local Refinement with Smooth Objective
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_local = minimize(
                smooth_objective, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            local_score = exact_score(res_local.x)
            if local_score > best_score:
                best_score = local_score
                best_x = res_local.x
        except Exception:
            pass

    # Fallback if DE failed or didn't improve significantly
    if best_x is None or best_score < 0.1:
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            x_start = np.random.uniform(-1, 1, (n * d))
            try:
                res = minimize(smooth_objective, x_start, method='L-BFGS-B', options={'maxiter': 50})
                s = exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x
            except Exception:
                continue

    if best_x is None:
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        best_x = np.column_stack((xx.ravel(), yy.ravel())).ravel()
        dmax_init = np.max(pdist(best_x.reshape(n, d)))
        if dmax_init > 1e-12:
            best_x = (best_x / dmax_init).ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
