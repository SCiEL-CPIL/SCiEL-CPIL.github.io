"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed where possible, but N=14 is small enough for simple loops.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Iterate over all pairs (i, j) and compute cross product with all k > j
    # This avoids redundant calculations and keeps memory low.
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # Vectors from i to all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            # Cross product magnitude gives 2 * area
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for minimization.
    We want to maximize score, so we minimize -score.
    """
    points = points_flat.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _soft_min_objective(points_flat, beta=100.0):
    """
    Objective function using a soft-min approximation for the triangle areas.
    This makes the objective smoother, potentially helping gradient-based methods
    or allowing for more stable convergence.
    """
    points = points_flat.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
    
    if hull_area < 1e-10:
        return 1e10
        
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas.append(tri_areas)
                
    if not areas:
        return 1e10
        
    all_areas = np.concatenate(areas)
    
    if len(all_areas) == 0 or np.any(all_areas < 1e-10):
        return 1e10
        
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, we can use log-sum-exp
    # log(sum(exp(-beta * areas))) = -beta * min(areas) + log(sum(exp(beta*(min(areas)-areas))))
    min_a = np.min(all_areas)
    # If min_a is very small, the soft min will be close to min_a
    # We want to maximize min_a / hull_area, so minimize -soft_min / hull_area
    
    # Using log-sum-exp trick for stability
    # Let z_i = -beta * areas_i
    # sum(exp(z_i)) = exp(max_z) * sum(exp(z_i - max_z))
    # log_sum = max_z + log(sum(exp(z_i - max_z)))
    
    z = -beta * all_areas
    max_z = np.max(z)
    if max_z < -1000: # Underflow risk, though exp(-1000) is tiny
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    
    score = soft_min_val / hull_area
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Multiple diverse initializations (regular polygon, jittered, etc.)
    2. For each init, run Nelder-Mead then Powell
    3. Track best result across all runs
    4. Final polish with tight tolerances
    5. Normalize the result to unit area
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    def _refine_from_start(x0, time_budget):
        """Refine a starting point using NM then Powell, returning best found."""
        best_x = x0.copy()
        best_s = _score(best_x.reshape(N, 2))
        
        # Nelder-Mead
        t_check = time.time()
        if t_check < deadline - max(5, time_budget * 0.4):
            try:
                res = minimize(
                    _objective, best_x, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                s = -res.fun
                if s > best_s:
                    best_s = s
                    best_x = res.x
            except Exception:
                pass
        
        # Powell
        t_check = time.time()
        if t_check < deadline - 5:
            try:
                res = minimize(
                    _objective, best_x, method='Powell',
                    options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                s = -res.fun
                if s > best_s:
                    best_s = s
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_s
    
    # Diverse initializations
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Regular 14-gon with small angular jitter (breaks symmetry)
    rng = np.random.default_rng(42)
    angles_j = angles + rng.normal(0, 0.05, N)
    inits.append(np.column_stack([np.cos(angles_j), np.sin(angles_j)]).flatten())
    
    # 3. Two concentric heptagons (7+7 split)
    r_inner, r_outer = 0.5, 1.0
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts_inner = np.column_stack([r_inner * np.cos(angles7), r_inner * np.sin(angles7)])
    angles7b = angles7 + np.pi / 7
    pts_outer = np.column_stack([r_outer * np.cos(angles7b), r_outer * np.sin(angles7b)])
    inits.append(np.vstack([pts_outer, pts_inner]).flatten())
    
    # 4. Jittered regular polygon with larger perturbation
    angles_j2 = angles + rng.normal(0, 0.15, N)
    inits.append(np.column_stack([np.cos(angles_j2), np.sin(angles_j2)]).flatten())
    
    # 5. Slightly squashed ellipse-like distribution
    angles_e = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([1.2 * np.cos(angles_e), 0.8 * np.sin(angles_e)]).flatten())
    
    # Run refinement on each initialization
    best_result = inits[0].copy()
    best_score_val = _score(best_result.reshape(N, 2))
    
    # Allocate time budget per init
    n_inits = len(inits)
    time_per_init = max(15.0, (deadline - start_time - 10) / n_inits)
    
    for idx, x0 in enumerate(inits):
        if time.time() >= deadline - 15:
            break
        
        t0 = time.time()
        x_refined, s_refined = _refine_from_start(x0, time_per_init)
        
        if s_refined > best_score_val:
            best_score_val = s_refined
            best_result = x_refined
        
        # Check elapsed and break if needed
        if time.time() >= deadline - 15:
            break
    
    # Final polish on the best result
    if time.time() < deadline - 8:
        try:
            res_final = minimize(
                _objective, best_result, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res_final.fun
            if s > best_score_val:
                best_score_val = s
                best_result = res_final.x
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            res_final2 = minimize(
                _objective, best_result, method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-11, 'ftol': 1e-13}
            )
            s = -res_final2.fun
            if s > best_score_val:
                best_score_val = s
                best_result = res_final2.x
        except Exception:
            pass
    
    # Final normalization
    final_points = best_result.reshape(N, 2)
    
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        final_points = np.column_stack([np.cos(angles), np.sin(angles)]).copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        final_points = np.column_stack([np.cos(angles), np.sin(angles)]).copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    
    return final_points
# EVOLVE-BLOCK-END
