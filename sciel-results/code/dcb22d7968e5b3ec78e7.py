# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(pts):
    """Return (d_min/d_max)^2, or 0 on degenerate input."""
    try:
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        dmin = d.min()
        dmax = d.max()
        if dmax <= 1e-12 or not np.isfinite(dmin) or not np.isfinite(dmax):
            return 0.0
        r = dmin / dmax
        return float(r * r)
    except Exception:
        return 0.0

def _build_points(params):
    """
    Build 14 points from 5 parameters:
    z_p: z-coordinate of pole 1 (pole 2 is -z_p)
    r_1, z_1: radius and z-height of ring 1 (6 points, phase 0)
    r_2, z_2: radius and z-height of ring 2 (6 points, phase pi/6)
    """
    z_p, r_1, z_1, r_2, z_2 = params
    points = np.zeros((14, 3))
    
    # Poles
    points[0] = [0.0, 0.0, z_p]
    points[1] = [0.0, 0.0, -z_p]
    
    # Ring 1: 6 points, phase 0
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for i in range(6):
        points[2 + i] = [r_1 * np.cos(angles1[i]), r_1 * np.sin(angles1[i]), z_1]
        
    # Ring 2: 6 points, phase pi/6 (staggered)
    angles2 = angles1 + np.pi / 6.0
    for i in range(6):
        points[8 + i] = [r_2 * np.cos(angles2[i]), r_2 * np.sin(angles2[i]), z_2]
        
    return points

def _neg_score_params(params):
    """Negative score for minimization."""
    pts = _build_points(params)
    return -_score(pts)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_pts = None
    best_val = -1.0
    
    # Define search ranges
    # z_p: 0 to 1.5 (symmetry handles sign, but we can just search positive and mirror if needed, 
    # but the build function uses +/- z_p, so z_p > 0 is sufficient for poles)
    # r_1, r_2: 0.1 to 1.5
    # z_1, z_2: -1.5 to 1.5
    
    # Initial coarse grid
    # We will search over z_p, r_1, z_1, r_2, z_2
    # To keep it fast, we'll do a nested loop or meshgrid for a subset, then refine.
    
    # Let's try a multi-stage approach.
    # Stage 1: Coarse grid
    z_p_vals = np.linspace(0.0, 1.2, 13)       # 13 steps
    r_1_vals = np.linspace(0.2, 1.2, 11)       # 11 steps
    z_1_vals = np.linspace(-0.8, 0.8, 17)      # 17 steps
    r_2_vals = np.linspace(0.2, 1.2, 11)       # 11 steps
    z_2_vals = np.linspace(-0.8, 0.8, 17)      # 17 steps
    
    # Total combos: 13 * 11 * 17 * 11 * 17 = ~450,000. 
    # 450k evaluations of 14-point pdist is fast in numpy? 
    # pdist on 14 points is 91 distances. 450k * 91 ~ 40M ops. Should be < 5-10 seconds.
    
    # We can vectorize the score calculation for the grid to be faster.
    # But _score is scalar. Let's write a vectorized batch scorer for the grid search.
    
    def _batch_score(params_array):
        """
        params_array: shape (N, 5)
        returns: array of scores (N,)
        """
        N = params_array.shape[0]
        scores = np.zeros(N)
        for i in range(N):
            if time.time() > deadline:
                break
            pts = _build_points(params_array[i])
            scores[i] = _score(pts)
        return scores

    # Generate full grid
    # This might be too big for a simple loop if not optimized. 
    # Let's use meshgrid and flatten.
    ZP, R1, Z1, R2, Z2 = np.meshgrid(z_p_vals, r_1_vals, z_1_vals, r_2_vals, z_2_vals, indexing='ij')
    grid_params = np.column_stack([ZP.ravel(), R1.ravel(), Z1.ravel(), R2.ravel(), Z2.ravel()])
    
    # Filter out invalid radii (though linspace starts at 0.2)
    # Evaluate in chunks to monitor time
    chunk_size = 5000
    best_param_coarse = None
    best_score_coarse = -1.0
    
    total_params = len(grid_params)
    for i in range(0, total_params, chunk_size):
        if time.time() > deadline - 60:
            break
        chunk = grid_params[i:i+chunk_size]
        scores = _batch_score(chunk)
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score_coarse:
            best_score_coarse = scores[max_idx]
            best_param_coarse = chunk[max_idx]
            
    if best_param_coarse is None:
        # Fallback if grid search failed or timed out immediately
        best_param_coarse = np.array([0.5, 0.8, 0.0, 0.8, 0.0])
        best_score_coarse = _score(_build_points(best_param_coarse))

    # Stage 2: Fine Grid around the best coarse candidate
    # Expand search space slightly to ensure we didn't miss the edge
    z_p_min = max(0.0, best_param_coarse[0] - 0.1)
    z_p_max = min(1.5, best_param_coarse[0] + 0.1)
    
    r_1_min = max(0.1, best_param_coarse[1] - 0.1)
    r_1_max = min(1.5, best_param_coarse[1] + 0.1)
    
    z_1_min = max(-1.5, best_param_coarse[2] - 0.1)
    z_1_max = min(1.5, best_param_coarse[2] + 0.1)
    
    r_2_min = max(0.1, best_param_coarse[3] - 0.1)
    r_2_max = min(1.5, best_param_coarse[3] + 0.1)
    
    z_2_min = max(-1.5, best_param_coarse[4] - 0.1)
    z_2_max = min(1.5, best_param_coarse[4] + 0.1)
    
    # Finer steps
    n_fine = 25
    z_p_fine = np.linspace(z_p_min, z_p_max, n_fine)
    r_1_fine = np.linspace(r_1_min, r_1_max, n_fine)
    z_1_fine = np.linspace(z_1_min, z_1_max, n_fine)
    r_2_fine = np.linspace(r_2_min, r_2_max, n_fine)
    z_2_fine = np.linspace(z_2_min, z_2_max, n_fine)
    
    ZPF, R1F, Z1F, R2F, Z2F = np.meshgrid(z_p_fine, r_1_fine, z_1_fine, r_2_fine, z_2_fine, indexing='ij')
    fine_params = np.column_stack([ZPF.ravel(), R1F.ravel(), Z1F.ravel(), R2F.ravel(), Z2F.ravel()])
    
    best_param_fine = best_param_coarse
    best_score_fine = best_score_coarse
    
    total_fine = len(fine_params)
    for i in range(0, total_fine, chunk_size):
        if time.time() > deadline - 30:
            break
        chunk = fine_params[i:i+chunk_size]
        scores = _batch_score(chunk)
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score_fine:
            best_score_fine = scores[max_idx]
            best_param_fine = chunk[max_idx]
            
    # Stage 3: Local Optimization (L-BFGS-B) on the 5D parameter space
    # Start from best_param_fine
    # We minimize -score
    try:
        # Bounds for L-BFGS-B
        bounds = [
            (0.0, 1.5),   # z_p
            (0.1, 1.5),   # r_1
            (-1.5, 1.5),  # z_1
            (0.1, 1.5),   # r_2
            (-1.5, 1.5)   # z_2
        ]
        
        # L-BFGS-B requires a smooth function. _score is piecewise (min/max of distances).
        # It is not differentiable at points where d_min changes identity.
        # However, Nelder-Mead or Powell might be more robust for non-smooth functions.
        # Let's use Nelder-Mead or Powell.
        
        res = minimize(
            _neg_score_params,
            best_param_fine,
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if res.fun < best_score_fine: # res.fun is negative
            best_score_fine = -res.fun
            best_param_fine = res.x
            
        # Try one more polish with Powell
        res2 = minimize(
            _neg_score_params,
            best_param_fine,
            method='Powell',
            options={
                'maxiter': 500,
                'xtol': 1e-10,
                'ftol': 1e-12
            }
        )
        if res2.fun < best_score_fine:
            best_score_fine = -res2.fun
            best_param_fine = res2.x
            
    except Exception:
        pass

    # Construct final points
    final_pts = _build_points(best_param_fine)
    final_score = _score(final_pts)
    
    # If the symmetry approach yielded a very low score, try a generic random restart fallback
    # This is a safety net in case the D3h symmetry is not the global optimum
    if final_score < 0.1:
        print("Symmetry approach low score, attempting fallback...")
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            # Random points on sphere
            pts = rng.normal(0, 1, (14, 3))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Local refine
            def _neg_score_flat(x):
                return -_score(x.reshape(14,3))
            
            res = minimize(
                _neg_score_flat,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True}
            )
            s = -res.fun
            if s > final_score:
                final_score = s
                final_pts = res.x.reshape(14, 3)

    # Final validation
    if not np.all(np.isfinite(final_pts)):
        # Absolute fallback
        final_pts = _build_points(np.array([0.5, 0.8, 0.0, 0.8, 0.0]))
        
    return final_pts

# EVOLVE-BLOCK-END
