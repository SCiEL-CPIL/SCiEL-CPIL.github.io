import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin

    def score_points(pts):
        """Compute (max_dist / min_dist)^2 for a set of points."""
        pts = pts - np.mean(pts, axis=0)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def objective_with_penalty(params):
        """
        Objective for optimization:
        Primary: (max_dist / min_dist)^2
        Secondary: Anisotropy penalty based on Gram matrix eigenvalues.
        """
        if time.time() > deadline - 2:
            return 1e10
        
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        
        # Primary score
        primary_score = score_points(pts)
        if primary_score > 1e9:
            return primary_score

        # Secondary: Anisotropy Penalty
        # Center the points
        pts_centered = pts - np.mean(pts, axis=0)
        # Gram Matrix
        G = pts_centered.T @ pts_centered
        # Eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(G)
            # Filter out very small eigenvalues to avoid division by zero
            eigvals = eigvals[eigvals > 1e-12]
            if len(eigvals) < 2:
                # Degenerate case, points are collinear or identical
                return primary_score + 100.0
            
            max_eig = np.max(eigvals)
            min_eig = np.min(eigvals)
            
            # Penalty is ratio of max to min eigenvalue
            # This penalizes elongated distributions
            anisotropy_penalty = max_eig / min_eig
        except Exception:
            anisotropy_penalty = 0.0

        # Weight for penalty: Small enough not to distort the primary objective too much
        # but enough to guide towards isotropy
        alpha = 0.1 
        return primary_score + alpha * anisotropy_penalty

    # 1. Generate Hexagonal Lattice
    R = 6
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    
    # Candidate pool for swaps
    pool_size = 150
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = pts[indices]
        return score_points(sub_pts)

    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    # 2. Greedy Swap Search on Lattice
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i_idx, i in enumerate(current_indices):
            if time.time() > deadline - 30:
                break
            for j in outside_candidates:
                if time.time() > deadline - 30:
                    break
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break
            if improved:
                break
                
    best_pts = pts[current_indices]
    best_score = current_score

    # 3. Continuous Refinement
    # Normalize: fix p0 to (0,0) and p1 to (1,0)
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1: 
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    # Align p0->(0,0), p1->(1,0)
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    # Reorder
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    bnds = [(-10, 10)] * (2 * (N - 2))
    
    best_params = init_params
    best_opt_score = score_points(best_pts) # Use raw score for comparison
    
    try:
        # Use L-BFGS-B for fast, deterministic refinement
        res = minimize(
            objective_with_penalty, 
            init_params, 
            method='L-BFGS-B', 
            bounds=bnds, 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Check if the optimized points are valid and better
        final_pts_opt = np.zeros((N, 2))
        final_pts_opt[0] = [0.0, 0.0]
        final_pts_opt[1] = [1.0, 0.0]
        final_pts_opt[2:] = res.x.reshape(-1, 2)
        
        # Check distinctness
        diff_opt = final_pts_opt[:, np.newaxis, :] - final_pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff_opt**2, axis=2))
        iu_opt = np.triu_indices(N, 1)
        d_opt = dists_opt[iu_opt]
        
        if np.min(d_opt) > 1e-6:
            raw_score_opt = score_points(final_pts_opt)
            if raw_score_opt < best_opt_score:
                best_opt_score = raw_score_opt
                best_params = res.x
                best_pts = final_pts_opt
    except Exception:
        pass

    # Final Polish without penalty to ensure we are at the local minimum of the raw objective
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                lambda p: score_points(np.vstack([[0,0],[1,0],p.reshape(-1,2)])), 
                best_params, 
                method='L-BFGS-B', 
                bounds=bnds, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            final_pts_polish = np.zeros((N, 2))
            final_pts_polish[0] = [0.0, 0.0]
            final_pts_polish[1] = [1.0, 0.0]
            final_pts_polish[2:] = res2.x.reshape(-1, 2)
            
            diff_pol = final_pts_polish[:, np.newaxis, :] - final_pts_polish[np.newaxis, :, :]
            dists_pol = np.sqrt(np.sum(diff_pol**2, axis=2))
            iu_pol = np.triu_indices(N, 1)
            d_pol = dists_pol[iu_pol]
            
            if np.min(d_pol) > 1e-6:
                raw_score_pol = score_points(final_pts_polish)
                if raw_score_pol < best_opt_score:
                    best_opt_score = raw_score_pol
                    best_pts = final_pts_polish
        except Exception:
            pass

    return best_pts
