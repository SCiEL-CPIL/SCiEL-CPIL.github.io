import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave margin

    def score_subset(indices):
        sub_pts = pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def smooth_objective(params, beta_max=200.0, beta_min=200.0):
        """
        Smooth surrogate for (max_d / min_d)^2 using log-sum-exp.
        soft_max = (1/beta) * log(sum(exp(beta * d)))
        soft_min = -(1/beta) * log(sum(exp(-beta * d)))
        """
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        # Avoid numerical issues
        d_opt = np.maximum(d_opt, 1e-10)
        
        # Soft max via log-sum-exp
        max_d_soft = (1.0 / beta_max) * np.log(np.sum(np.exp(beta_max * d_opt)))
        
        # Soft min via -log-sum-exp of negative
        min_d_soft = -(1.0 / beta_min) * np.log(np.sum(np.exp(-beta_min * d_opt)))
        
        if min_d_soft < 1e-8:
            return 1e10
        return (max_d_soft / min_d_soft)**2

    def exact_objective(params):
        """Exact objective for final evaluation."""
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

    def normalize_points(pts_in):
        """
        Normalize points: fix p0->(0,0), p1->(1,0) where (p0,p1) is the closest pair.
        Returns (init_params, perm) where init_params are the free coordinates.
        """
        diff = pts_in[:, np.newaxis, :] - pts_in[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = pts_in[i0]
        p1 = pts_in[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            # Fallback: use first two points
            i0, i1 = 0, 1
            p0 = pts_in[0]
            p1 = pts_in[1]
            v = p1 - p0
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                return None, None
        
        # Align p0->(0,0), p1->(1,0)
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts_in - p0) @ R_mat.T / norm_v
        
        # Reorder
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        return init_params, perm

    def optimize_from_init(init_params, bnds, time_limit):
        """Run smooth optimization then exact polish."""
        best_params = init_params.copy()
        best_score = exact_objective(best_params)
        
        # Phase 1: Smooth optimization with L-BFGS-B
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < best_score:
                best_params = res.x
                best_score = exact_objective(best_params)
        except Exception:
            pass
        
        if time.time() > deadline - time_limit:
            return best_params, best_score
        
        # Phase 2: Polish with exact objective and tighter tolerance
        try:
            res2 = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass
        
        return best_params, best_score

    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Triangular lattice cluster
    R = 6
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    lattice_pts = np.array(lattice_pts)
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_configs.append(lattice_pts[sorted_indices[:N]])
    
    # Config 2: Circular pattern (16 points on a circle + center)
    circle_pts = np.zeros((N, 2))
    circle_pts[0] = [0, 0]
    for i in range(1, N):
        angle = 2 * np.pi * (i - 1) / (N - 1)
        circle_pts[i] = [np.cos(angle), np.sin(angle)]
    initial_configs.append(circle_pts)
    
    # Config 3: Two concentric circles (8 inner, 8 outer)
    circle2_pts = np.zeros((N, 2))
    for i in range(8):
        angle = 2 * np.pi * i / 8
        circle2_pts[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        circle2_pts[8 + i] = [1.0 * np.cos(angle), 1.0 * np.sin(angle)]
    initial_configs.append(circle2_pts)
    
    # Config 4: Square grid 4x4
    grid_pts = np.zeros((N, 2))
    for i in range(4):
        for j in range(4):
            grid_pts[4*i + j] = [i, j]
    initial_configs.append(grid_pts)
    
    # Config 5: Triangular arrangement (rows of 4,4,4,4)
    tri_pts = np.zeros((N, 2))
    row_len = 4
    for row in range(4):
        for col in range(row_len):
            idx = row * row_len + col
            tri_pts[idx] = [col + 0.5 * row, row * np.sqrt(3) / 2]
    initial_configs.append(tri_pts)
    
    bnds = [(-10, 10)] * (2 * (N - 2))
    
    best_params = None
    best_score = float('inf')
    
    # Time budget for initial optimizations
    n_configs = len(initial_configs)
    time_per_config = (deadline - start_time - 10) / n_configs  # Leave 10s margin
    
    for cfg_idx, cfg in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # Normalize the configuration
        init_params, perm = normalize_points(cfg)
        if init_params is None:
            continue
        
        # Check if we have time
        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        
        # Optimize from this initial configuration
        params, score = optimize_from_init(init_params, bnds, time_per_config)
        
        if score < best_score:
            best_score = score
            best_params = params
    
    # If we didn't get any valid result, fallback to lattice
    if best_params is None:
        init_params, perm = normalize_points(initial_configs[0])
        if init_params is not None:
            best_params = init_params
            best_score = exact_objective(best_params)
        else:
            # Last resort: return the lattice points
            return initial_configs[0]
    
    # Final high-precision polish if we have time
    if time.time() < deadline - 5:
        try:
            res_final = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                                 options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_final.fun < best_score:
                best_params = res_final.x
                best_score = res_final.fun
        except Exception:
            pass
    
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
