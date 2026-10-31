import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    # 150s limit, leave 5s margin for safety
    deadline = start_time + 145 

    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        min_d = np.min(d_opt)
        max_d = np.max(d_opt)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Shift for numerical stability
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Soft max: log(sum(exp(beta * d_sq))) / beta
        # Use max_d_sq to prevent overflow
        exp_terms_max = np.exp(beta * (d_sq - max_d_sq))
        max_d_sq_soft = max_d_sq + np.log(np.sum(exp_terms_max)) / beta
        
        # Soft min: -log(sum(exp(-beta * d_sq))) / beta
        # Use min_d_sq to prevent overflow in exp(-beta * d_sq) if d_sq is large? 
        # Actually, exp(-beta * (d_sq - min_d_sq)) is safe if d_sq >= min_d_sq
        exp_terms_min = np.exp(-beta * (d_sq - min_d_sq))
        min_d_sq_soft = min_d_sq - np.log(np.sum(exp_terms_min)) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    def normalize_and_parametrize(pts):
        """Normalize points so min distance = 1, align closest pair to (0,0) and (1,0)"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        if min_d < 1e-8:
            return None
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None
        
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    # --- Phase 1: Lattice-based Initialization and Swap Search ---
    # Generate a triangular lattice
    R = 8
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    
    # Sort by distance to origin to get a compact initial set
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    # Initial subset: 16 closest points
    initial_subset = sorted_indices[:N]
    
    # Candidate pool: Top 200 closest points
    pool_size = 200
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = lattice_pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    # Greedy Swap Search
    improved = True
    while improved and time.time() < deadline - 20: # Reserve 20s for continuous opt
        improved = False
        # Shuffle outside candidates to avoid bias
        rng = np.random.RandomState(int(time.time()*1000) % 10000)
        rng.shuffle(outside_candidates)
        
        for i_idx, i in enumerate(current_indices):
            if time.time() > deadline - 20:
                break
            for j in outside_candidates:
                if time.time() > deadline - 20:
                    break
                # Swap i and j
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break # Restart j loop
            if improved:
                break # Restart i loop

    # Convert best lattice subset to parameters
    best_pts = lattice_pts[current_indices]
    best_params = normalize_and_parametrize(best_pts)
    
    # --- Phase 2: Continuous Optimization ---
    bnds = [(-10, 10)] * (2 * (N - 2))
    
    best_params_opt = None
    best_score_opt = 1e10
    
    if best_params is not None:
        current_params = best_params.copy()
        current_score_opt = exact_objective(current_params)
        
        # If lattice solution is bad, try other initializations
        if current_score_opt > 2.0:
             # Fallback to random or other geometric configs if lattice failed
             rng = np.random.RandomState(42)
             rand_pts = rng.uniform(-3, 3, size=(N, 2))
             # Ensure min dist > 0.1
             diff = rand_pts[:, np.newaxis, :] - rand_pts[np.newaxis, :, :]
             dists = np.sqrt(np.sum(diff**2, axis=2))
             if np.min(dists[np.triu_indices(N, 1)]) > 0.1:
                 alt_params = normalize_and_parametrize(rand_pts)
                 if alt_params is not None:
                     current_params = alt_params
                     current_score_opt = exact_objective(current_params)

        # Multi-start continuous optimization
        # We will run a few rounds of optimization on the current best, 
        # and potentially perturb it.
        
        # 1. Soft Surrogate Optimization (Adaptive Beta)
        for beta in [20.0, 40.0, 80.0]:
            if time.time() > deadline - 10:
                break
            try:
                def soft_obj_adaptive(params, b=beta):
                    return soft_objective(params, beta=b)
                
                res = minimize(soft_obj_adaptive, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < current_score_opt:
                        current_params = res.x.copy()
                        current_score_opt = exact_score
            except Exception:
                pass
        
        # 2. Exact L-BFGS-B
        if time.time() < deadline - 8:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < current_score_opt:
                    current_params = res.x.copy()
                    current_score_opt = res.fun
            except Exception:
                pass
        
        # 3. Powell for non-smoothness
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, current_params, method='Powell', bounds=bnds,
                               options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-8})
                if res.fun < current_score_opt:
                    current_params = res.x.copy()
                    current_score_opt = res.fun
            except Exception:
                pass
        
        # 4. Perturbation and Re-optimize (Escape local minima)
        rng = np.random.RandomState(123)
        for perturb_iter in range(10):
            if time.time() > deadline - 3:
                break
            # Perturb best solution
            perturbation_scale = 0.2 / (1 + perturb_iter * 0.2)
            perturbed = current_params + rng.normal(0, perturbation_scale, size=current_params.shape)
            for i in range(len(perturbed)):
                perturbed[i] = np.clip(perturbed[i], bnds[i][0], bnds[i][1])
            
            perturbed_score = exact_objective(perturbed)
            
            # Quick local optimization from perturbed point
            try:
                res = minimize(exact_objective, perturbed, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < current_score_opt:
                    current_score_opt = res.fun
                    current_params = res.x.copy()
            except Exception:
                pass
        
        best_params_opt = current_params
        best_score_opt = current_score_opt

    # --- Phase 3: Final Polish ---
    if best_params_opt is not None and time.time() < deadline - 1:
        try:
            res = minimize(exact_objective, best_params_opt, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score_opt:
                best_params_opt = res.x
                best_score_opt = res.fun
        except Exception:
            pass

    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    if best_params_opt is not None:
        final_pts[2:] = best_params_opt.reshape(-1, 2)
    else:
        # Fallback to lattice points if optimization failed completely
        final_pts = best_pts

    # Verify no duplicate points
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists_final = np.sqrt(np.sum(diff**2, axis=2))
    iu_final = np.triu_indices(N, 1)
    d_final = dists_final[iu_final]
    if np.min(d_final) < 1e-10:
        # Fallback to simple lattice
        return lattice_pts[sorted_indices[:N]]

    return final_pts

# EVOLVE-BLOCK-END
