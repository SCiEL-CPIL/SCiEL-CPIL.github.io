import numpy as np
import scipy.spatial as spat
import time
from scipy.optimize import minimize, differential_evolution

N = 14

def calc_score(points):
    """
    Calculate min triangle area / convex hull area.
    Returns 0 if hull is degenerate.
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    n = len(points)
    
    # Pre-computed indices for C(14,3) = 364 triangles
    # This is fast enough for local optimization steps
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Global Search using Differential Evolution
    # We search in a normalized box [-1, 1] x [-1, 1].
    # The score is scale invariant, so the absolute scale doesn't matter, 
    # but keeping points in a bounded box helps DE.
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Objective function for DE (minimize negative score)
    def neg_score_flat(x):
        if time.time() > deadline - 10:
            return 0.0
        pts = x.reshape(N, 2)
        # Ensure points are not all identical or collinear
        if np.ptp(pts, axis=0).max() < 1e-6:
            return 0.0
        return -calc_score(pts)
    
    best_global = None
    best_global_score = 0.0
    
    # Run DE with a limited budget
    # maxiter is a soft limit, but we check time in the objective
    # popsize=10, N=28 -> 280 evaluations per generation.
    # We can afford roughly 1000-2000 evaluations in ~30-40s.
    # So ~10-20 generations.
    
    try:
        # seed=42 for reproducibility
        res_de = differential_evolution(
            neg_score_flat, 
            bounds, 
            seed=42, 
            maxiter=20, 
            popsize=15,
            tol=0.01,
            init='sobol',
            polish=False # We will do our own polish
        )
        
        if time.time() < deadline - 10:
            best_global = res_de.x
            best_global_score = -res_de.fun
    except Exception:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_global = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
        best_global_score = calc_score(best_global.reshape(N, 2))

    # 2. Local Refinement using L-BFGS-B
    # Start from the best global candidate and a few other good candidates
    # (e.g., regular polygon, slightly perturbed regular polygon)
    
    candidates = []
    if best_global is not None:
        candidates.append(best_global)
    
    # Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    candidates.append(reg_pts.flatten())
    
    # Two heptagons (current best strategy's starting point)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_outer = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    pts_inner = 0.8 * np.column_stack((np.cos(base_angles + np.pi/7), np.sin(base_angles + np.pi/7)))
    hept_pts = np.vstack((pts_outer, pts_inner))
    candidates.append(hept_pts.flatten())
    
    # Remove duplicates (simple check)
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.allclose(c, uc, atol=1e-5):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    
    best_local = None
    best_local_score = 0.0
    
    for x0 in unique_candidates:
        if time.time() > deadline - 15:
            break
        
        # L-BFGS-B requires a differentiable objective.
        # calc_score is continuous but not differentiable where min triangle changes.
        # However, L-BFGS-B often works well on such "min of smooth functions" landscapes
        # if we use a soft-min or just rely on the quasi-Newton approximation.
        # To be safer, we can use a smoothed minimum: -logsumexp(-areas) approximates min.
        # But for high precision, direct minimization of -min_area is often sufficient 
        # if we start close enough.
        
        def neg_score_local(x):
            if time.time() > deadline - 5:
                return 0.0
            pts = x.reshape(N, 2)
            return -calc_score(pts)
        
        try:
            res_min = minimize(
                neg_score_local, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() < deadline - 5:
                local_score = -res_min.fun
                if local_score > best_local_score:
                    best_local_score = local_score
                    best_local = res_min.x
        except Exception:
            pass

    # 3. Final Polish with Nelder-Mead (robust to non-differentiability)
    # Run from the best local result and the best global result
    
    final_candidates = []
    if best_local is not None:
        final_candidates.append(best_local)
    if best_global is not None:
        final_candidates.append(best_global)
    
    best_final = None
    best_final_score = 0.0
    
    for x0 in final_candidates:
        if time.time() > deadline - 5:
            break
        
        def neg_score_nm(x):
            if time.time() > deadline - 1:
                return 0.0
            pts = x.reshape(N, 2)
            return -calc_score(pts)
        
        try:
            res_nm = minimize(
                neg_score_nm, 
                x0, 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200}
            )
            
            if time.time() < deadline - 1:
                nm_score = -res_nm.fun
                if nm_score > best_final_score:
                    best_final_score = nm_score
                    best_final = res_nm.x
        except Exception:
            pass

    # Fallback if all optimizations failed
    if best_final is None:
        best_final = reg_pts.flatten()
    
    return best_final.reshape(N, 2)
