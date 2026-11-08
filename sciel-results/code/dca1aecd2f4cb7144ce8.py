import numpy as np
from scipy.optimize import minimize
import time
import math

N = 21

def _epigraph_objective(params, W, H, beta):
    """Epigraph formulation: maximize sum of t_i subject to t_i <= soft_min(dists_i).
    
    params = [x_1, y_1, ..., x_N, y_N, t_1, t_2, ..., t_N]
    We minimize -sum(t_i) subject to t_i <= soft_min(distances_i)
    
    For smoothness, we use: t_i + (1/beta) * log(sum(exp(beta*(d_j - t_i)))) <= 0
    which is equivalent to t_i <= soft_min(d_j) in the smooth sense.
    """
    n = N
    centers = params[:2*n].reshape(n, 2)
    t = params[2*n:]
    
    # Objective: minimize -sum(t)
    obj = -np.sum(t)
    
    # Constraint: for each i, t_i <= soft_min of distances to walls and half-distances to others
    # soft_min(d) = min(d) - (1/beta)*log(sum(exp(-beta*(d - min(d)))))
    # Equivalent: t_i + (1/beta)*log(sum(exp(beta*(d_j - t_i)))) <= 0
    # But this is complex. Instead, let's use a penalty approach.
    
    # Actually, let's just return the objective for L-BFGS-B with bounds on t
    # The constraints will be handled by the bounds: t_i in [0, max_possible]
    # And we'll add a penalty for violating t_i > soft_min(dist_i)
    
    penalty = 0.0
    for i in range(n):
        d_walls = np.array([centers[i, 0], W - centers[i,0],
                           centers[i, 1], H - centers[i,1]])
        dists = np.linalg.norm(centers[i] - centers, axis=1) / 2.0
        dists[i] = np.inf
        all_dists = np.concatenate([d_walls, dists])
        
        # soft_min
        m = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - m))
        soft_min_val = m - (1.0/beta) * np.log(np.sum(exp_terms))
        
        # Penalty if t_i > soft_min_val
        if t[i] > soft_min_val:
            penalty += (t[i] - soft_min_val)**2
    
    return obj + penalty * 1e4

def _voronoi_inradii(centers, W, H):
    """Compute the inradius of each circle's Voronoi cell (vectorized)."""
    n = len(centers)
    # Distance to walls for all circles at once
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    # Half-distance to other circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # For each circle, min of wall dists and inter-circle half-dists
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    radii = np.min(all_dists, axis=1)
    return radii

def _soft_min(arr, beta):
    """Numerically stable soft-min approximation."""
    m = np.min(arr)
    exp_terms = np.exp(-beta * (arr - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _soft_min_sum(centers, W, H, beta):
    """Compute sum of soft-min approximations of radii for all circles (vectorized)."""
    n = len(centers)
    # Distance to walls for all circles
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    # Half-distance to other circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    
    # Soft-min for each circle
    mins = np.min(all_dists, axis=1)
    exp_terms = np.exp(-beta * (all_dists - mins[:, np.newaxis]))
    log_sums = np.log(np.sum(exp_terms, axis=1))
    soft_mins = mins - (1.0 / beta) * log_sums
    
    return np.sum(soft_mins)

def _objective(centers_flat, W, H, beta):
    """Negative sum of soft-min radii (to minimize)."""
    centers = centers_flat.reshape(N, 2)
    return -_soft_min_sum(centers, W, H, beta)

def _seed_hex(W, H, seed):
    """Generate initial center positions using a hexagonal-like pattern."""
    np.random.seed(seed)
    
    # Hexagonal packing: rows offset
    # Estimate rows and columns
    aspect = W / H
    if aspect >= 1:
        n_cols = max(2, int(np.sqrt(N * aspect)))
        n_rows = max(1, int(np.ceil(N / n_cols)))
    else:
        n_rows = max(2, int(np.sqrt(N / aspect)))
        n_cols = max(1, int(np.ceil(N / n_rows)))
    
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(centers) >= N:
                break
            x = (col + 0.5) * W / n_cols
            y = (row + 0.5) * H / n_rows
            # Offset every other row for hexagonal pattern
            if row % 2 == 1:
                x += W / (2 * n_cols)
            x = np.clip(x, 0.05, W - 0.05)
            y = np.clip(y, 0.05, H - 0.05)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # If we don't have enough, add random points
    if len(centers) < N:
        extra = np.random.uniform(0.05, W - 0.05, (N - len(centers), 2))
        extra[:, 1] = np.random.uniform(0.05, H - 0.05, N - len(centers))
        centers = np.vstack([centers, extra])
    
    # Small random perturbation to break symmetry
    centers += np.random.normal(0, 0.02, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    return centers

def _seed_grid(W, H, seed):
    """Generate initial center positions using a grid with perturbation."""
    np.random.seed(seed)
    n_x = max(2, int(np.sqrt(N * W / H)))
    n_y = max(1, int(np.ceil(N / n_x)))
    
    x_coords = np.linspace(0.1, W - 0.1, n_x + 2)[1:-1] if n_x > 1 else np.array([W/2])
    y_coords = np.linspace(0.1, H - 0.1, n_y + 2)[1:-1] if n_y > 1 else np.array([H/2])
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    centers = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    if len(centers) < N:
        extra_x = np.linspace(0.1, W - 0.1, N - len(centers) + 1)
        extra_y = np.linspace(0.1, H - 0.1, N - len(centers) + 1)
        extra = np.column_stack((extra_x, extra_y))
        centers = np.vstack([centers, extra])
    
    centers = centers[:N]
    centers += np.random.normal(0, 0.03, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    return centers

def _optimize(W, H, initial_centers, deadline):
    """Multi-stage optimization with increasing beta and final high-precision polish."""
    bounds = [(0, W)] * N + [(0, H)] * N
    
    current_centers = initial_centers.copy()
    best_centers = current_centers.copy()
    best_score = np.sum(_voronoi_inradii(current_centers, W, H))
    
    # Multi-stage beta schedule
    beta_schedule = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0]
    max_iters_schedule = [150, 200, 250, 200, 200, 150]
    ftol_schedule = [1e-10, 1e-12, 1e-12, 1e-14, 1e-14, 1e-14]
    gtol_schedule = [1e-8, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12]
    
    for stage, (beta, max_iter, ftol, gtol) in enumerate(
        zip(beta_schedule, max_iters_schedule, ftol_schedule, gtol_schedule)
    ):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        try:
            res = minimize(
                _objective, current_centers.flatten(),
                args=(W, H, beta),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            current_centers = res.x.reshape(N, 2)
        except Exception:
            break
        
        # Evaluate exact score
        score = np.sum(_voronoi_inradii(current_centers, W, H))
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
    
    # Final high-precision polish: use very high beta with tight tolerances
    if time.time() < deadline - 3:
        try:
            beta_final = 800.0
            res_final = minimize(
                _objective, best_centers.flatten(),
                args=(W, H, beta_final),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            final_centers = res_final.x.reshape(N, 2)
            final_score = np.sum(_voronoi_inradii(final_centers, W, H))
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_score = -np.inf
    best_config = None

    # Search over rectangle widths (perimeter = 4 => H = 2 - W)
    # Include more width values for better coverage
    widths = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    # Use both grid and hexagonal seeds
    seed_funcs = [_seed_grid, _seed_hex]
    
    n_restarts_per_seed = 5

    for W in widths:
        if time.time() > deadline:
            break
        H = 2.0 - W

        for seed_func in seed_funcs:
            for restart in range(n_restarts_per_seed):
                if time.time() > deadline - 10:  # Leave 10s for final evaluations
                    break

                seed = 42 + restart * 10 + (0 if seed_func == _seed_grid else 5)
                centers = seed_func(W, H, seed=seed)
                
                try:
                    final_centers, score = _optimize(W, H, centers, deadline)
                    
                    if score > best_score:
                        best_score = score
                        radii = _voronoi_inradii(final_centers, W, H)
                        best_config = np.column_stack((final_centers, radii))
                        
                except Exception:
                    pass

    # Fallback if no valid result
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
