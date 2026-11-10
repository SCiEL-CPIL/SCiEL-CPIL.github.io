import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centers):
    """
    Compute the maximum common radius for a set of circle centers.
    Returns 0 if any center is outside or if there are coincident centers (dist=0).
    """
    # Check bounds
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances using broadcasting (faster than cdist for small N)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Only look at upper triangle
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu[0], iu[1]].min() / 2.0
    
    return min(min_boundary, min_pair)

def _compute_radius_active(centers, active_pairs=None):
    """
    Compute radius, optionally only considering a subset of pairs.
    If active_pairs is None, compute all pairs.
    """
    if np.any(np.abs(centers) > 0.5):
        return 0.0, None, None
    
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0, None, None
    
    if active_pairs is None:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        iu = np.triu_indices(N, k=1)
        min_pair = dists[iu[0], iu[1]].min() / 2.0
        # Find which pairs are at minimum
        all_half_dists = dists[iu[0], iu[1]] / 2.0
        min_val = all_half_dists.min()
        active_mask = all_half_dists <= min_val + 1e-12
        active_pairs = list(zip(iu[0][active_mask], iu[1][active_mask]))
    else:
        min_pair = np.inf
        for i, j in active_pairs:
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_pair:
                min_pair = d
    
    r = min(min_boundary, min_pair)
    return r, min_boundary, min_pair

def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of min radius using log-sum-exp.
    This is differentiable and suitable for L-BFGS-B.
    """
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Compute all pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    half_dists = dists / 2.0
    min_d = half_dists.min()
    
    # Soft min: -log(sum(exp(-beta*x)))/beta
    # Use log-sum-exp trick for stability
    exp_vals = np.exp(-beta * (half_dists - min_d))
    soft_min_dist = min_d - np.log(exp_vals.sum()) / beta
    
    return min(min_boundary, soft_min_dist)

def _hex_init(n):
    """Create a hexagonal grid initialization."""
    best_centers = None
    best_score = -1
    
    # Try different row/col configurations
    for rows in range(4, 14):
        cols = int(np.ceil(n / rows))
        if cols < 3 or cols > 15:
            continue
        
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        x_start = -0.5 + dx / 2
        y_start = -0.5 + dy / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = x_start + j * dx
                y = y_start + i * dy
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        if len(centers) >= n:
            centers = np.array(centers[:n])
            centers = np.clip(centers, -0.5, 0.5)
            s = _compute_radius(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers

def _targeted_local_search(centers, time_budget=5.0):
    """
    Targeted local search that only adjusts circles involved in minimum-distance pairs.
    This is much faster than full-space optimization.
    """
    start_time = time.time()
    current = centers.copy()
    current_r, min_b, min_p = _compute_radius_active(current)
    
    if current_r <= 0:
        return current
    
    max_iterations = 200
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Get active pairs
        r, min_boundary, min_pair = _compute_radius_active(current)
        if r <= 0:
            break
        
        # Determine which constraint is binding
        if min_pair <= min_boundary:
            # Pairwise constraint is binding
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            iu = np.triu_indices(N, k=1)
            all_half_dists = dists[iu[0], iu[1]] / 2.0
            min_val = all_half_dists.min()
            active_mask = all_half_dists <= min_val + 1e-10
            active_pairs = list(zip(iu[0][active_mask], iu[1][active_mask]))
            
            if not active_pairs:
                break
            
            # For each active pair, try to push them apart
            improved = False
            for i, j in active_pairs:
                # Try small perturbations to increase distance
                d_vec = current[i] - current[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm < 1e-12:
                    d_dir = np.array([1.0, 0.0])
                else:
                    d_dir = d_vec / d_norm
                
                # Try moving i in direction d_dir and j in -d_dir
                for scale in [0.001, 0.002, 0.005, 0.01]:
                    test_i = current[i].copy()
                    test_j = current[j].copy()
                    test_i += scale * d_dir
                    test_j -= scale * d_dir
                    
                    # Check bounds
                    if np.any(np.abs(test_i) > 0.5) or np.any(np.abs(test_j) > 0.5):
                        continue
                    
                    current[i] = test_i
                    current[j] = test_j
                    new_r, _, _ = _compute_radius_active(current)
                    
                    if new_r > r + 1e-12:
                        r = new_r
                        improved = True
                        break
                    else:
                        # Revert
                        current[i] = test_i - scale * d_dir
                        current[j] = test_j + scale * d_dir
                
                if improved:
                    break
            else:
                # No improvement from pairwise moves, try boundary moves
                # Find circles closest to boundary
                boundary = 0.5 - np.max(np.abs(current), axis=1)
                min_b_idx = np.argmin(boundary)
                for scale in [0.001, 0.002, 0.005]:
                    # Move away from boundary
                    c = current[min_b_idx]
                    # Determine direction to move inward
                    move_dir = -np.sign(c) * 0.001
                    for axis in range(2):
                        if abs(c[axis]) == np.max(np.abs(c)):
                            move_dir = np.zeros(2)
                            move_dir[axis] = -np.sign(c[axis]) * scale
                            break
                    test_c = c + move_dir
                    if np.any(np.abs(test_c) > 0.5):
                        continue
                    current[min_b_idx] = test_c
                    new_r, _, _ = _compute_radius_active(current)
                    if new_r > r + 1e-12:
                        r = new_r
                        improved = True
                        break
                    else:
                        current[min_b_idx] = c
                
                if not improved:
                    break
        else:
            # Boundary constraint is binding
            boundary = 0.5 - np.max(np.abs(current), axis=1)
            min_b_idx = np.argmin(boundary)
            c = current[min_b_idx]
            
            # Move inward
            for scale in [0.001, 0.002, 0.005]:
                # Determine direction to move inward
                move_dir = np.zeros(2)
                for axis in range(2):
                    if abs(c[axis]) == np.max(np.abs(c)):
                        move_dir[axis] = -np.sign(c[axis]) * scale
                        break
                test_c = c + move_dir
                if np.any(np.abs(test_c) > 0.5):
                    continue
                current[min_b_idx] = test_c
                new_r, _, _ = _compute_radius_active(current)
                if new_r > r + 1e-12:
                    r = new_r
                    break
                else:
                    current[min_b_idx] = c
            else:
                break
    
    return current

def _optimize_smooth(centers, time_budget):
    """
    Optimize using L-BFGS-B on the soft-min radius.
    """
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        # Maximize soft radius = minimize negative soft radius
        r = _soft_min_radius(centers_2d, beta=100.0)
        return -r
    
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 300,
            'ftol': 1e-14,
            'gtol': 1e-8,
            'maxls': 50
        }
    )
    
    polished = result.x.reshape(-1, 2)
    polished = np.clip(polished, -0.5, 0.5)
    return polished

def _optimize_exact(centers, time_budget):
    """
    Optimize using Nelder-Mead on the exact (nonsmooth) radius.
    """
    start_time = time.time()
    
    def penalized_objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        if np.any(np.abs(centers_2d) > 0.5):
            return 1000.0 # Large penalty
        r = _compute_radius(centers_2d)
        return -r
        
    result = minimize(
        penalized_objective,
        centers.flatten(),
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-12
        }
    )
    
    polished = result.x.reshape(-1, 2)
    polished = np.clip(polished, -0.5, 0.5)
    return polished

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1
    
    # 1. Initialization
    centers = _hex_init(N)
    score = _compute_radius(centers)
    if score > best_score:
        best_centers = centers.copy()
        best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 2. Targeted Local Search (fast, focuses on active constraints)
    if remaining > 80:
        tl_result = _targeted_local_search(best_centers, time_budget=min(10, remaining - 5))
        tl_score = _compute_radius(tl_result)
        if tl_score > best_score:
            best_centers = tl_result
            best_score = tl_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 3. Smooth Optimization (L-BFGS-B)
    # This helps escape the grid structure and find a better basin.
    if remaining > 50:
        smooth_result = _optimize_smooth(best_centers, time_budget=min(25, remaining - 5))
        smooth_score = _compute_radius(smooth_result)
        if smooth_score > best_score:
            best_centers = smooth_result
            best_score = smooth_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 4. Another targeted local search after smooth optimization
    if remaining > 40:
        tl_result2 = _targeted_local_search(best_centers, time_budget=min(10, remaining - 5))
        tl_score2 = _compute_radius(tl_result2)
        if tl_score2 > best_score:
            best_centers = tl_result2
            best_score = tl_score2
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 5. Exact Optimization (Nelder-Mead)
    # This polishes the solution to the exact max-min value.
    if remaining > 25:
        exact_result = _optimize_exact(best_centers, time_budget=min(15, remaining - 5))
        exact_score = _compute_radius(exact_result)
        if exact_score > best_score:
            best_centers = exact_result
            best_score = exact_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 6. Final targeted polish
    if remaining > 15:
        final_tl = _targeted_local_search(best_centers, time_budget=min(8, remaining - 5))
        final_tl_score = _compute_radius(final_tl)
        if final_tl_score > best_score:
            best_centers = final_tl
            best_score = final_tl_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 7. Multi-Restart Local Search
    # Try a few perturbations and re-optimize to escape local minima.
    for restart in range(4):
        if time.time() - start_time > total_time_limit - 8:
            break
        
        # Perturb best centers
        perturbation_scale = 0.003 / (restart + 1)
        perturbed = best_centers + np.random.randn(N, 2) * perturbation_scale
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Quick targeted local search
        if remaining > 12:
            tl_res = _targeted_local_search(perturbed, time_budget=min(5, remaining - 5))
            t_score = _compute_radius(tl_res)
            if t_score > best_score:
                best_centers = tl_res
                best_score = t_score
        
        # Quick smooth optimization if time permits
        if remaining > 8:
            smooth_res = _optimize_smooth(best_centers, time_budget=min(4, remaining - 5))
            s_score = _compute_radius(smooth_res)
            if s_score > best_score:
                best_centers = smooth_res
                best_score = s_score
        
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation check
    final_r = _compute_radius(best_centers)
    if final_r <= 0 or not np.all(np.isfinite(best_centers)):
        best_centers = _hex_init(N)
    
    return best_centers
