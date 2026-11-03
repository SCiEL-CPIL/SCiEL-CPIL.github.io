import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(10):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.90 + 0.05 * attempt)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.03, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair = np.sqrt(sq_dists.min()) / 2.0
    return min(min_pair, min_bound)

def _feasibility_objective_and_grad(x, r_target):
    """
    Objective: Minimize sum of squared violations for a target radius r_target.
    E(x) = sum_{i<j} max(0, 2*r_target - d_ij)^2 + sum_i max(0, r_target + ||c_i|| - 1)^2
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    
    # Objective value
    # Pairwise violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - d)
    # We only need i < j to avoid double counting in sum, but for gradient we need full matrix or careful indexing
    # Let's compute full sum then divide by 2 for pairs? No, the term is for each pair (i,j).
    # If we sum over all i != j, we count each pair twice.
    # Let's just sum over upper triangle for objective.
    iu = np.triu_indices(n, k=1)
    pair_viol_upper = pair_viol[iu]
    obj_pairs = np.sum(pair_viol_upper ** 2)
    
    # Boundary violations
    bound_viol = np.maximum(0.0, r_target + norms - 1.0)
    obj_bounds = np.sum(bound_viol ** 2)
    
    obj = obj_pairs + obj_bounds
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    # d/dx_i (max(0, 2r - d_ij)^2)
    # If d_ij < 2r:
    #   2 * (2r - d_ij) * d/dx_i (2r - d_ij)
    #   d/dx_i d_ij = (x_i - x_j) / d_ij
    #   So term is 2 * (2r - d_ij) * (x_i - x_j) / d_ij
    # Note: This is the force on i due to j.
    
    # We can compute the gradient vector for each point i as:
    # grad_pairs[i] = sum_{j != i} 2 * (2r - d_ij) * (x_i - x_j) / d_ij  (if d_ij < 2r)
    
    # To do this efficiently:
    # Create a mask for violations
    mask_pair = (d < 2.0 * r_target) & (d > 1e-10) # Avoid div by zero
    # Compute weights W_ij = 2 * (2r - d_ij) / d_ij
    # Where mask is true, else 0
    W = np.zeros((n, n))
    W[mask_pair] = 2.0 * (2.0 * r_target - d[mask_pair]) / d[mask_pair]
    
    # grad_pairs[i] = sum_j W_ij * (x_i - x_j)
    # = sum_j W_ij * x_i - sum_j W_ij * x_j
    # = x_i * sum_j W_ij - (W @ x)_i
    
    W_sum = W.sum(axis=1, keepdims=True)
    grad_pairs = W_sum * pts - (W @ pts)
    
    # Boundary gradient
    # d/dx_i (max(0, r + ||x_i|| - 1)^2)
    # If r + ||x_i|| > 1:
    #   2 * (r + ||x_i|| - 1) * d/dx_i (||x_i||)
    #   d/dx_i ||x_i|| = x_i / ||x_i||
    #   So term is 2 * (r + ||x_i|| - 1) * x_i / ||x_i||
    
    mask_bound = (norms > 1.0 - r_target) & (norms > 1e-10)
    W_bound = np.zeros(n)
    W_bound[mask_bound] = 2.0 * (r_target + norms[mask_bound] - 1.0) / norms[mask_bound]
    
    grad_bounds = W_bound[:, np.newaxis] * pts
    
    grad = (grad_pairs + grad_bounds).ravel()
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initialization
    best_pts = _hex_init(N, seed=42)
    best_r = _compute_radius(best_pts)
    
    # Try a few seeds to find a better start
    for seed in [0, 1, 2, 3, 5]:
        if time.time() > deadline - 100.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    x0 = best_pts.ravel()
    r_current = best_r
    
    # 2. Iterative Radius Increase
    # We will try to increase r_current by small steps.
    # If we fail, we try to "kick" the configuration.
    
    max_iter = 200
    kick_count = 0
    max_kicks = 50
    
    for i_iter in range(max_iter):
        if time.time() > deadline - 20.0:
            break
            
        # Determine step size
        # Start with 0.1%, decrease if stuck
        step_factor = 0.001
        if kick_count > 0:
            step_factor = 0.0005
        if kick_count > 10:
            step_factor = 0.0001
            
        r_target = r_current * (1.0 + step_factor)
        
        # Optimize to satisfy constraints for r_target
        try:
            res = minimize(
                lambda x: _feasibility_objective_and_grad(x, r_target)[0],
                x0,
                jac=lambda x: _feasibility_objective_and_grad(x, r_target)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                    'maxfun': 500,
                }
            )
            
            new_pts = res.x.reshape(-1, 2)
            # Check validity
            norms = np.linalg.norm(new_pts, axis=1)
            if not np.all(np.isfinite(new_pts)) or np.any(norms >= 1.0):
                # If invalid, try to clip
                mask = norms >= 1.0
                if np.any(mask):
                    new_pts[mask] *= 0.999 / norms[mask, np.newaxis]
                    
            new_r = _compute_radius(new_pts)
            
            # Check if we actually achieved the target radius (or close to it)
            # The objective value tells us how much violation there is.
            # If obj is small, r_target is feasible.
            obj_val = res.fun
            
            if obj_val < 1e-6 and new_r >= r_target * 0.999:
                # Success
                r_current = r_target
                x0 = new_pts.ravel()
                best_r = new_r
                best_pts = new_pts.copy()
                kick_count = 0
            else:
                # Failure to reach target
                kick_count += 1
                
                if kick_count > max_kicks:
                    # Stuck, try a larger kick or break
                    # Try a random perturbation
                    rng = np.random.default_rng(i_iter)
                    noise = rng.normal(0, 0.01, x0.shape)
                    x0_noisy = x0 + noise
                    # Re-normalize to stay inside
                    pts_noisy = x0_noisy.reshape(-1, 2)
                    norms_noisy = np.linalg.norm(pts_noisy, axis=1)
                    mask_noisy = norms_noisy >= 1.0
                    if np.any(mask_noisy):
                        pts_noisy[mask_noisy] *= 0.999 / norms_noisy[mask_noisy, np.newaxis]
                    x0 = pts_noisy.ravel()
                    kick_count = 0
                else:
                    # Reduce step size and try again with same x0
                    # Or try a smaller r_target
                    pass
                    
        except Exception as e:
            # If optimization fails, kick
            kick_count += 1
            if kick_count > max_kicks:
                rng = np.random.default_rng(i_iter)
                noise = rng.normal(0, 0.01, x0.shape)
                x0_noisy = x0 + noise
                pts_noisy = x0_noisy.reshape(-1, 2)
                norms_noisy = np.linalg.norm(pts_noisy, axis=1)
                mask_noisy = norms_noisy >= 1.0
                if np.any(mask_noisy):
                    pts_noisy[mask_noisy] *= 0.999 / norms_noisy[mask_noisy, np.newaxis]
                x0 = pts_noisy.ravel()
                kick_count = 0
            continue

    # 3. Final Polish
    # Use the soft-min objective to fine-tune the final configuration
    if time.time() < deadline - 5.0:
        try:
            # Re-implement soft min objective for final polish
            def _soft_min_obj_grad(x, T=0.001):
                pts = x.reshape(-1, 2)
                n = len(pts)
                norms = np.linalg.norm(pts, axis=1)
                boundary = 1.0 - norms
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                d = np.linalg.norm(diff, axis=2)
                iu = np.triu_indices(n, k=1)
                pair_half = d[iu] / 2.0
                all_vals = np.concatenate([pair_half, boundary])
                
                neg_vals_T = -all_vals / T
                max_nv = np.max(neg_vals_T)
                exp_shifted = np.exp(neg_vals_T - max_nv)
                weights = exp_shifted / np.sum(exp_shifted)
                obj = -T * (np.log(np.sum(exp_shifted)) + max_nv)
                
                # Gradient
                grad_all = -weights
                safe_dists = np.where(d > 1e-10, d, 1.0)
                unit_vecs = diff / safe_dists[:, :, np.newaxis]
                W = np.zeros((n, n))
                W[iu] = grad_all[:len(pair_half)]
                W[iu[1], iu[0]] = grad_all[:len(pair_half)]
                grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
                
                safe_norms = np.where(norms > 1e-10, norms, 1.0)
                grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
                
                grad = (grad_pairs + grad_boundary).ravel()
                return obj, grad

            res = minimize(
                lambda x: _soft_min_obj_grad(x)[0],
                x0,
                jac=lambda x: _soft_min_obj_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                }
            )
            new_pts = res.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(np.isfinite(new_pts)) and np.all(norms < 1.0):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    return best_pts
