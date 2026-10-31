"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    # Generate a hexagonal lattice with enough points
    R = 4
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    
    def score_points(pts):
        # pts is an array of shape (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Local search: try swapping one point in the subset with one outside
    # Use a broader candidate pool for better initial configuration
    candidate_indices = sorted_indices[:60].tolist()
    
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    max_local_iter = 200
    local_iter = 0
    while improved and time.time() < deadline - 40 and local_iter < max_local_iter:
        improved = False
        local_iter += 1
        for i in range(N):
            if time.time() > deadline - 40:
                break
            for j in outside:
                if time.time() > deadline - 40:
                    break
                new_subset = initial_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    initial_subset = new_subset
                    in_subset = set(initial_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break  # restart outer loop after improvement
    
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # Refine with continuous optimization
    # Fix the first two points to remove translation and rotation invariance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair[0], pair[1]
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    # Rotation matrix
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform all points
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder so that i0 is at index 0 and i1 is at index 1
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    init_params = transformed[2:].flatten()
    
    def soft_objective(params, beta=100.0):
        """Smooth surrogate using log-sum-exp for max and -log-sum-exp for min."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Avoid zero distances
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max of distances (in squared space)
        # Use log-sum-exp trick for numerical stability
        max_d_sq = np.max(d_sq)
        soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        # Soft min of distances (in squared space)
        min_d_sq = np.min(d_sq)
        soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min_d_sq < 1e-8:
            return 1e10
        
        # We want to minimize (max_d / min_d)^2 = max_d_sq / min_d_sq
        return soft_max_d_sq / soft_min_d_sq
    
    def epigraph_objective(params, t=1.0):
        """Epigraph formulation: minimize t subject to d_sq[i] <= t*min_d_sq for all i.
        This is equivalent but may have smoother gradients near the optimum."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        d_sq = np.maximum(d_sq, 1e-12)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-8:
            return 1e10
        
        return max_d_sq / min_d_sq
    
    def exact_objective(params):
        """Exact objective for final evaluation."""
        if time.time() > deadline:
            return 1e10
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_d = np.min(d)
        max_d = np.max(d)
        
        if min_d < 1e-8:
            return 1e10
        
        return (max_d / min_d)**2
    
    bnds = [(-6, 6)] * 28
    
    best_params = init_params.copy()
    best_exact_score = exact_objective(init_params)
    
    # Phase 1: Optimize smooth surrogate with adaptive beta
    try:
        if time.time() < deadline - 35:
            # Start with lower beta for smoother landscape, then increase
            for beta_val in [20.0, 50.0, 100.0, 200.0, 400.0]:
                if time.time() > deadline - 35:
                    break
                res = minimize(soft_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10},
                               args=(beta_val,))
                
                candidate_score = exact_objective(res.x)
                if candidate_score < best_exact_score:
                    best_params = res.x
                    best_exact_score = candidate_score
                        
    except Exception:
        pass
    
    # Phase 2: Polish with exact objective from best smooth solution
    try:
        if time.time() < deadline - 20:
            res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            candidate_score = exact_objective(res2.x)
            if candidate_score < best_exact_score:
                best_params = res2.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    # Phase 2b: Nelder-Mead polish (handles non-smooth min/max better)
    try:
        if time.time() < deadline - 12:
            nm_init = best_params.copy()
            res_nm = minimize(exact_objective, nm_init, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            candidate_score = exact_objective(res_nm.x)
            if candidate_score < best_exact_score:
                best_params = res_nm.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    # Phase 3: Multiple restarts with moderate perturbations to escape local minima
    try:
        if time.time() < deadline - 10:
            rng = np.random.default_rng(42)
            for restart in range(12):
                if time.time() > deadline - 10:
                    break
                    
                # Vary perturbation scale: small for fine-tuning, larger for escape
                if restart < 4:
                    scale = 0.03
                elif restart < 8:
                    scale = 0.10
                else:
                    scale = 0.25
                perturbation = rng.normal(0, scale, size=28)
                perturbed_params = best_params + perturbation
                perturbed_params = np.clip(perturbed_params, -6, 6)
                
                # Use smooth objective for restarts, then polish
                res3 = minimize(soft_objective, perturbed_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-10},
                               args=(100.0,))
                
                candidate_score = exact_objective(res3.x)
                if candidate_score < best_exact_score:
                    best_params = res3.x
                    best_exact_score = candidate_score
                    # Quick NM polish after improvement
                    if time.time() < deadline - 10:
                        res_nm2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                        cand2 = exact_objective(res_nm2.x)
                        if cand2 < best_exact_score:
                            best_params = res_nm2.x
                            best_exact_score = cand2
                        
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    try:
        if time.time() < deadline - 5:
            res4 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14})
            
            candidate_score = exact_objective(res4.x)
            if candidate_score < best_exact_score:
                best_params = res4.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    # Phase 5: Final ultra-precise Nelder-Mead polish if time permits
    try:
        if time.time() < deadline - 2:
            res5 = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            
            candidate_score = exact_objective(res5.x)
            if candidate_score < best_exact_score:
                best_params = res5.x
                best_exact_score = candidate_score
                    
    except Exception:
        pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify final score
    final_score = score_points(final_pts)
    if final_score < best_score:
        return final_pts
    else:
        return best_pts

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
