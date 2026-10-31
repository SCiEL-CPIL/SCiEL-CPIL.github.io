import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    def compute_ratio(pts):
        """Compute squared max/min distance ratio."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def normalize_and_align(pts):
        """Normalize points so closest pair is at (0,0) and (1,0)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            return pts, pts[2:].flatten(), 0, 1
        
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
        
        return transformed, transformed[2:].flatten(), i0, i1

    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        return compute_ratio(pts_opt)

    def soft_objective(params, beta=30.0):
        """Smooth surrogate using log-sum-exp."""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Numerically stable log-sum-exp
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Soft max
        log_sum_exp_max = max_d_sq + np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        max_d_sq_soft = log_sum_exp_max / beta
        
        # Soft min
        log_sum_exp_min = min_d_sq + np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        min_d_sq_soft = -log_sum_exp_min / beta + 2*min_d_sq  # correction
        
        # Actually let's do it properly:
        # soft_min = -1/beta * log(sum(exp(-beta * x)))
        # = -1/beta * (min(x) + log(sum(exp(-beta*(x - min(x))))))
        # = -min(x) - 1/beta * log(sum(exp(-beta*(x - min(x)))))
        
        min_val = np.min(d_sq)
        log_sum_exp_min = min_val + np.log(np.sum(np.exp(-beta * (d_sq - min_val))))
        min_d_sq_soft = -(min_val + np.log(np.sum(np.exp(-beta * (d_sq - min_val)))) / beta)
        
        if min_d_sq_soft < 1e-10:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    def epigraph_objective(params, t):
        """Epigraph formulation: minimize t subject to all d^2 <= t^2 * min_d^2."""
        # This is complex; use simpler approach
        pass

    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Config 1: Triangular lattice - closest 16 to origin
    R = 8
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    dists_origin = np.sum(lattice_pts**2, axis=1)
    sorted_idx = np.argsort(dists_origin)
    init1 = lattice_pts[sorted_idx[:N]]
    initial_configs.append(init1)
    
    # Config 2: Regular 4x4 grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append((i - 1.5, j - 1.5))
    grid_pts = np.array(grid_pts)
    initial_configs.append(grid_pts)
    
    # Config 3: Two concentric circles (8 inner, 8 outer)
    circle_pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        circle_pts.append((0.5 * np.cos(angle), 0.5 * np.sin(angle)))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        circle_pts.append((1.5 * np.cos(angle), 1.5 * np.sin(angle)))
    circle_pts = np.array(circle_pts)
    initial_configs.append(circle_pts)
    
    # Config 4: Hexagonal arrangement
    hex_pts = []
    # Center
    hex_pts.append((0, 0))
    # Inner ring (6)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        hex_pts.append((0.8 * np.cos(angle), 0.8 * np.sin(angle)))
    # Outer ring (9)
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9
        hex_pts.append((1.8 * np.cos(angle), 1.8 * np.sin(angle)))
    hex_pts = np.array(hex_pts)
    initial_configs.append(hex_pts)
    
    # Config 5: Lattice subset with local search
    pool_size = 200
    candidate_indices = sorted_idx[:pool_size]
    current_indices = sorted_idx[:N].copy()
    current_score = compute_ratio(lattice_pts[current_indices])
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 40:
        improved = False
        for i_idx in range(N):
            if time.time() > deadline - 40:
                break
            for j in outside_candidates:
                if time.time() > deadline - 40:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = compute_ratio(lattice_pts[new_indices])
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
    
    initial_configs.append(lattice_pts[current_indices])
    
    # Optimize each initial configuration
    best_params = None
    best_score = 1e10
    
    for config_idx, config in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
        
        aligned, params, i0, i1 = normalize_and_align(config)
        current_score = exact_objective(params)
        
        if current_score < best_score:
            best_params = params.copy()
            best_score = current_score
        
        bnds = [(-10, 10)] * (2 * (N - 2))
        
        # Phase 1: Soft surrogate
        if time.time() < deadline - 20:
            try:
                res = minimize(soft_objective, params, method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e9:
                    exact_s = exact_objective(res.x)
                    if exact_s < best_score:
                        best_params = res.x.copy()
                        best_score = exact_s
                        params = res.x.copy()
            except Exception:
                pass
        
        # Phase 2: Exact L-BFGS-B
        if time.time() < deadline - 10:
            try:
                res = minimize(exact_objective, params, method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < best_score:
                    best_params = res.x.copy()
                    best_score = res.fun
                    params = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 5:
            try:
                res = minimize(exact_objective, params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_score:
                    best_params = res.x.copy()
                    best_score = res.fun
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res = minimize(exact_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_params = res.x.copy()
                best_score = res.fun
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
