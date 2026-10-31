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
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        exp_terms_max = np.exp(beta * (d_sq - max_d_sq))
        max_d_sq_soft = max_d_sq + np.log(np.sum(exp_terms_max)) / beta
        
        exp_terms_min = np.exp(-beta * (d_sq - min_d_sq))
        min_d_sq_soft = min_d_sq - np.log(np.sum(exp_terms_min)) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    def epigraph_objective(params, beta=30.0):
        """Epigraph formulation: minimize t subject to max_d^2 <= t*min_d^2, using soft approximations"""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Soft max and min of squared distances
        exp_terms_max = np.exp(beta * (d_sq - max_d_sq))
        max_d_sq_soft = max_d_sq + np.log(np.sum(exp_terms_max)) / beta
        
        exp_terms_min = np.exp(-beta * (d_sq - min_d_sq))
        min_d_sq_soft = min_d_sq - np.log(np.sum(exp_terms_min)) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        ratio_sq = max_d_sq_soft / min_d_sq_soft
        return ratio_sq

    def normalize_and_parametrize(pts):
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

    def score_subset(indices, lattice_pts):
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

    # --- Phase 1: Multiple diverse initializations ---
    initializations = []
    
    # 1a. Triangular lattice
    R = 10
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    pool_size = 300
    candidate_indices = sorted_indices[:pool_size]
    
    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices, lattice_pts)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        rng = np.random.RandomState(int(time.time()*1000) % 10000)
        rng.shuffle(outside_candidates)
        
        for i_idx in range(N):
            if time.time() > deadline - 25:
                break
            for j in outside_candidates:
                if time.time() > deadline - 25:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices, lattice_pts)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
    
    best_pts = lattice_pts[current_indices]
    best_params = normalize_and_parametrize(best_pts)
    if best_params is not None:
        initializations.append((best_params, exact_objective(best_params)))
    
    # 1b. Circular arrangement
    circle_pts = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        circle_pts[i] = [np.cos(angle), np.sin(angle)]
    circle_params = normalize_and_parametrize(circle_pts)
    if circle_params is not None:
        initializations.append((circle_params, exact_objective(circle_params)))
    
    # 1c. Two concentric circles
    two_circle_pts = np.zeros((N, 2))
    n_inner = 8
    n_outer = 8
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner
        two_circle_pts[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer + np.pi / n_outer
        two_circle_pts[n_inner + i] = [1.5 * np.cos(angle), 1.5 * np.sin(angle)]
    two_circle_params = normalize_and_parametrize(two_circle_pts)
    if two_circle_params is not None:
        initializations.append((two_circle_params, exact_objective(two_circle_params)))
    
    # 1d. Fibonacci spiral
    fib_pts = np.zeros((N, 2))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(N):
        r = np.sqrt(i + 1) * 0.3
        theta = golden_angle * (i + 1)
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    fib_params = normalize_and_parametrize(fib_pts)
    if fib_params is not None:
        initializations.append((fib_params, exact_objective(fib_params)))
    
    # 1e. Grid
    grid_pts = np.zeros((N, 2))
    idx = 0
    for i in range(4):
        for j in range(4):
            grid_pts[idx] = [i - 1.5, j - 1.5]
            idx += 1
    grid_params = normalize_and_parametrize(grid_pts)
    if grid_params is not None:
        initializations.append((grid_params, exact_objective(grid_params)))
    
    # 1f. Random with minimum distance constraint
    rng = np.random.RandomState(42)
    for attempt in range(5):
        rand_pts = rng.uniform(-3, 3, size=(N, 2))
        diff = rand_pts[:, np.newaxis, :] - rand_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        if np.min(dists[np.triu_indices(N, 1)]) > 0.3:
            rand_params = normalize_and_parametrize(rand_pts)
            if rand_params is not None:
                initializations.append((rand_params, exact_objective(rand_params)))
    
    # Select best initialization
    if not initializations:
        return best_pts
    
    initializations.sort(key=lambda x: x[1])
    current_params = initializations[0][0].copy()
    current_score_opt = initializations[0][1]
    
    # --- Phase 2: Continuous Optimization with multiple strategies ---
    bnds = [(-10, 10)] * (2 * (N - 2))
    
    best_params_opt = current_params.copy()
    best_score_opt = current_score_opt
    
    # Try each good initialization with optimization
    for init_idx, (init_params, init_score) in enumerate(initializations[:4]):
        if time.time() > deadline - 15:
            break
        if init_score >= best_score_opt * 1.5:  # Skip if much worse
            continue
        
        current_params = init_params.copy()
        current_score_opt = init_score
        
        # 2a. Epigraph-style soft objective (smoother than direct ratio)
        for beta in [20.0, 40.0, 60.0]:
            if time.time() > deadline - 10:
                break
            try:
                def soft_obj_adaptive(params, b=beta):
                    return epigraph_objective(params, beta=b)
                
                res = minimize(soft_obj_adaptive, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < current_score_opt:
                        current_params = res.x.copy()
                        current_score_opt = exact_score
            except Exception:
                pass
        
        # 2b. Direct soft objective
        for beta in [30.0, 50.0]:
            if time.time() > deadline - 8:
                break
            try:
                def soft_obj_direct(params, b=beta):
                    return soft_objective(params, beta=b)
                
                res = minimize(soft_obj_direct, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < current_score_opt:
                        current_params = res.x.copy()
                        current_score_opt = exact_score
            except Exception:
                pass
        
        # 2c. Exact L-BFGS-B
        if time.time() < deadline - 6:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-9})
                if res.fun < current_score_opt:
                    current_params = res.x.copy()
                    current_score_opt = res.fun
            except Exception:
                pass
        
        # 2d. Powell for non-smoothness
        if time.time() < deadline - 4:
            try:
                res = minimize(exact_objective, current_params, method='Powell', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-9})
                if res.fun < current_score_opt:
                    current_params = res.x.copy()
                    current_score_opt = res.fun
            except Exception:
                pass
        
        if current_score_opt < best_score_opt:
            best_score_opt = current_score_opt
            best_params_opt = current_params.copy()
    
    # 2e. Perturbation and re-optimize from best
    if time.time() < deadline - 3:
        rng = np.random.RandomState(123)
        for perturb_iter in range(15):
            if time.time() > deadline - 2:
                break
            perturbation_scale = 0.15 / (1 + perturb_iter * 0.15)
            perturbed = best_params_opt + rng.normal(0, perturbation_scale, size=best_params_opt.shape)
            for i in range(len(perturbed)):
                perturbed[i] = np.clip(perturbed[i], bnds[i][0], bnds[i][1])
            
            try:
                res = minimize(exact_objective, perturbed, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_score_opt:
                    best_score_opt = res.fun
                    best_params_opt = res.x.copy()
            except Exception:
                pass
    
    # --- Phase 3: Final high-precision polish ---
    if best_params_opt is not None and time.time() < deadline - 1:
        try:
            res = minimize(exact_objective, best_params_opt, method='L-BFGS-B', bounds=bnds, 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_score_opt:
                best_params_opt = res.x
                best_score_opt = res.fun
        except Exception:
            pass
        
        # Second polish with Powell
        if time.time() < deadline - 0.5:
            try:
                res = minimize(exact_objective, best_params_opt, method='Powell', bounds=bnds,
                               options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
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
        final_pts = best_pts

    # Verify no duplicate points
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists_final = np.sqrt(np.sum(diff**2, axis=2))
    iu_final = np.triu_indices(N, 1)
    d_final = dists_final[iu_final]
    if np.min(d_final) < 1e-10:
        return best_pts

    return final_pts

# EVOLVE-BLOCK-END
