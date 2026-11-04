import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Center the points to remove translation invariance
    centroid = np.mean(pts, axis=0)
    pts_c = pts - centroid
    
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts_c)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Check if all points are on the hull (convex position)
    # If not, we penalize heavily to steer towards convex configurations
    # The optimal solution for Heilbronn's problem is convex.
    if len(hull.vertices) < N:
        # Return a small negative score or 0 to discourage non-convex sets
        # Actually, just returning 0 might be enough if the convex score is positive.
        # But to be safe, we can return a very small number.
        # However, DE might get stuck in non-convex local minima if the score is 0.
        # Let's return the score but with a penalty.
        pass 

    # Compute all triangle areas vectorized
    idx_i, idx_j, idx_k = np.indices((N, N, N)).reshape(3, -1)
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i = idx_i[mask]
    j = idx_j[mask]
    k = idx_k[mask]
    
    p_i = pts_c[i]
    p_j = pts_c[j]
    p_k = pts_c[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(params):
    """
    Objective function for minimization.
    Returns negative score.
    """
    pts = params.reshape(N, 2)
    
    # Normalization: Center and scale to unit area hull?
    # No, just centering is enough for translation invariance.
    # Scaling invariance is handled by the ratio.
    
    score = _compute_score(pts)
    
    # If the points are not in convex position, we want to penalize them.
    # A non-convex set of 13 points will have a hull with < 13 vertices.
    # The minimum triangle area will likely be smaller than in the convex case.
    # However, to ensure the optimizer stays in the convex regime, we can add a penalty.
    try:
        hull = ConvexHull(pts)
        if len(hull.vertices) < N:
            # Penalty proportional to how "non-convex" it is?
            # Simpler: just return a bad score.
            # The score itself (min_area/hull_area) will be low for non-convex.
            # Let's trust the score.
            pass
    except:
        return 1e9

    if score == 0.0:
        return 1e9
        
    return -score

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin for 150s limit
    
    # Bounds for coordinates
    # We can normalize to [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Differential Evolution
    # maxiter and popsize control the search effort
    # seed for reproducibility
    try:
        de_res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=25, 
            tol=1e-8, 
            seed=42,
            polish=True, # Use local optimization (Nelder-Mead) at the end
            init='sobol' # Good space-filling sequence for initialization
        )
    except Exception as e:
        # Fallback to a simple heuristic if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        return pts

    best_pts = de_res.x.reshape(N, 2)
    
    # Check time
    if (time.time() - start_time) > time_limit:
        return best_pts

    # Additional local refinement with Nelder-Mead
    # DE's polish is already Nelder-Mead, but we can run it again with tighter tolerance
    # or try Powell
    try:
        nm_res = minimize(
            _objective, 
            best_pts, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if nm_res.fun < de_res.fun:
            best_pts = nm_res.x
    except Exception:
        pass
    
    # One more pass with Powell for differentiability-free local search
    if (time.time() - start_time) < time_limit - 10:
        try:
            pw_res = minimize(
                _objective, 
                best_pts, 
                method='Powell', 
                options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if pw_res.fun < _objective(best_pts):
                best_pts = pw_res.x
        except Exception:
            pass

    return best_pts
