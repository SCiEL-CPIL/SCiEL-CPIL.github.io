# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _surrogate_objective(x, p=4, alpha=2.0):
    """
    Smooth surrogate for the min/max distance ratio.
    We want to maximize d_min/d_max.
    Equivalently, minimize (d_max/d_min) or similar.
    
    A common smooth approximation for min is the negative power mean:
    min_i d_i ≈ - ( (1/n) sum (-d_i)^p )^(1/p) ... this is tricky with negatives.
    
    Instead, let's use:
    Score = (d_min / d_max)^2
    
    We want to maximize this. So we minimize -Score.
    But min and max are not differentiable.
    
    Approximation:
    d_min ≈ ( (1/N) sum d_i^(-p) )^(-1/p)  for large p
    d_max ≈ ( (1/N) sum d_i^p )^(1/p)      for large p
    
    Ratio ≈ [ ( (1/N) sum d_i^(-p) )^(-1/p) ] / [ ( (1/N) sum d_i^p )^(1/p) ]
          = [ (N sum d_i^(-p))^(-1/p) ] / [ (N sum d_i^p)^(1/p) ]  ... wait
    
    Let S_neg = sum d_i^(-p)
    Let S_pos = sum d_i^p
    
    d_min_approx = (S_neg / N)^(-1/p) = (N/S_neg)^(1/p)
    d_max_approx = (S_pos / N)^(1/p)
    
    Ratio = d_min_approx / d_max_approx = (N/S_neg)^(1/p) / (S_pos/N)^(1/p)
          = ( N^2 / (S_neg * S_pos) )^(1/p)
    
    We want to maximize Ratio^2 = ( N^2 / (S_neg * S_pos) )^(2/p)
    This is equivalent to minimizing (S_neg * S_pos)^(2/p)
    Or minimizing S_neg * S_pos.
    
    Let's use objective = S_neg * S_pos.
    """
    points = x.reshape(-1, 2)
    d = pdist(points)
    # Avoid division by zero
    d = np.maximum(d, 1e-10)
    
    S_neg = np.sum(d ** (-p))
    S_pos = np.sum(d ** p)
    
    return S_neg * S_pos


def _local_refine(points, max_iter=2000, tol=1e-8):
    """
    Refine points using gradient-free optimization (Nelder-Mead) on the surrogate.
    """
    x0 = points.flatten()
    n = len(points)
    
    def obj(x):
        return _surrogate_objective(x, p=8)
    
    # Use Nelder-Mead
    res = minimize(obj, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_points = res.x.reshape(-1, 2)
    return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Bounds for differential evolution
    # We can assume points are roughly in [-1, 1] or similar range.
    # The scale doesn't matter for the ratio, but DE needs bounds.
    bounds = [(-5, 5)] * (n * d)
    
    best_points = None
    best_score = -1.0
    
    # Run Differential Evolution with multiple restarts or a single long run
    # DE is stochastic, so we can try a few different seeds or just one robust run.
    # Given the 600s limit, we can afford a moderately sized DE run.
    
    # Strategy:
    # 1. Run DE to find a good global configuration.
    # 2. Refine with local optimization.
    # 3. If time permits, run DE again with different parameters or restarts.
    
    rng = np.random.default_rng(42)
    
    # First DE run
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(
                lambda x: _surrogate_objective(x, p=8),
                bounds=bounds,
                seed=42,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                polish=True  # Use L-BFGS-B at the end? Might be slow. Let's try False first and refine manually.
            )
            # Actually, polish uses L-BFGS-B which might struggle with the non-smooth surrogate if p is too high.
            # Let's rely on our own _local_refine.
            candidate_points = res.x.reshape(-1, 2)
            score = _compute_score(candidate_points)
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Local refinement
    if best_points is not None and time.time() < deadline - 30:
        refined_points = _local_refine(best_points, max_iter=1000)
        score = _compute_score(refined_points)
        if score > best_score:
            best_score = score
            best_points = refined_points.copy()
    
    # If we have time, try a second DE run with different parameters
    if time.time() < deadline - 90:
        try:
            res2 = differential_evolution(
                lambda x: _surrogate_objective(x, p=6),
                bounds=bounds,
                seed=123,
                maxiter=800,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            candidate_points = res2.x.reshape(-1, 2)
            score = _compute_score(candidate_points)
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
                
            if best_points is not None and time.time() < deadline - 30:
                refined_points = _local_refine(best_points, max_iter=1000)
                score = _compute_score(refined_points)
                if score > best_score:
                    best_score = score
                    best_points = refined_points.copy()
        except Exception:
            pass
            
    # If we have even more time, try a third run
    if time.time() < deadline - 60:
        try:
            res3 = differential_evolution(
                lambda x: _surrogate_objective(x, p=10),
                bounds=bounds,
                seed=777,
                maxiter=600,
                popsize=10,
                tol=1e-5,
                polish=False
            )
            candidate_points = res3.x.reshape(-1, 2)
            score = _compute_score(candidate_points)
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
                
            if best_points is not None and time.time() < deadline - 30:
                refined_points = _local_refine(best_points, max_iter=1000)
                score = _compute_score(refined_points)
                if score > best_score:
                    best_score = score
                    best_points = refined_points.copy()
        except Exception:
            pass
    
    # Fallback if DE failed
    if best_points is None:
        # Simple grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        grid = (grid - grid.mean(axis=0)) / grid.std()
        best_points = grid
    
    # Final check
    if time.time() < deadline - 10:
        final_refined = _local_refine(best_points, max_iter=500)
        score = _compute_score(final_refined)
        if score > best_score:
            best_points = final_refined
    
    return best_points


# EVOLVE-BLOCK-END
