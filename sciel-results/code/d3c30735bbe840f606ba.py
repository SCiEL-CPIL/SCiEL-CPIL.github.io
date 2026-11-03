import numpy as np
import scipy.optimize
import time
import math

N = 200

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        """Compute the maximum common radius for given centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < -1e-9:
            return 0.0
        dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def _soft_min_neg(values, beta):
        """
        Compute -soft_min(values, beta) and its gradient.
        soft_min(v, beta) = -1/beta * log(sum(exp(-beta * v)))
        -soft_min(v, beta) = 1/beta * log(sum(exp(-beta * v)))
        Gradient wrt v_i: -exp(-beta * v_i) / sum(exp(-beta * v_j))
        """
        # For numerical stability
        v = -beta * values
        max_v = np.max(v)
        exp_v = np.exp(v - max_v)
        sum_exp = np.sum(exp_v)
        val = (max_v + np.log(sum_exp)) / beta  # This is -soft_min
        # Gradient: d/dv_i (1/beta * log(sum(exp(-beta * v)))) 
        # = 1/beta * (1/sum) * (-beta * exp(-beta * v_i))
        # = -exp(-beta * v_i) / sum(exp(-beta * v_j))
        grad = -exp_v / sum_exp
        return val, grad

    def objective_and_grad(flat_params, beta):
        """
        Objective: minimize -r, where r is approximated by soft-min of constraints.
        Constraints:
        1. Wall distances: w_i = 0.5 - |x_i|, 0.5 - |y_i|
        2. Pair distances: d_ij = ||c_i - c_j|| / 2
        r_soft = soft_min(all_constraints, beta)
        We want to maximize r_soft, so minimize -r_soft.
        """
        centers = flat_params.reshape(N, 2)
        
        # Wall constraints
        # For each point, distance to 4 walls
        # w_left = x + 0.5, w_right = 0.5 - x, w_bottom = y + 0.5, w_top = 0.5 - y
        # We can just use min of these, but for soft min we need all of them?
        # Actually, the constraint is min_k (1/2 - |c_ik|).
        # Let's collect all wall distances.
        wx = np.array([centers[:, 0] + 0.5, 0.5 - centers[:, 0], 
                       centers[:, 1] + 0.5, 0.5 - centers[:, 1]])
        wall_dists = wx.flatten()
        
        # Pair constraints
        # We need all pair distances. This is O(N^2) memory if we store all, but N=200 -> 20k pairs, manageable.
        # dist_matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        # We only need unique pairs, but soft min over all is fine, just redundant.
        # To save computation, let's take upper triangle.
        idx_i, idx_j = np.triu_indices(N, k=1)
        pair_dists = dists[idx_i, idx_j] / 2.0
        
        # Combine all constraints
        all_constraints = np.concatenate([wall_dists, pair_dists])
        
        # Compute soft min
        soft_min_val, grad_constraints = _soft_min_neg(all_constraints, beta)
        
        # Objective is -soft_min_val (we want to maximize soft_min_val)
        # Wait, _soft_min_neg returns -soft_min.
        # So if f = -soft_min, we minimize f.
        # Let's clarify:
        # soft_min(v) approx min(v).
        # We want max(soft_min(v)).
        # So minimize -soft_min(v).
        # _soft_min_neg returns val = -soft_min(v) and grad of val.
        # So objective = val.
        
        obj = soft_min_val
        
        # Now map gradient from constraints to centers
        grad_centers = np.zeros_like(centers)
        
        # Wall gradients
        # wall_dists structure: [w_left_all, w_right_all, w_bottom_all, w_top_all]
        # w_left = x + 0.5 -> dx = 1
        # w_right = 0.5 - x -> dx = -1
        # w_bottom = y + 0.5 -> dy = 1
        # w_top = 0.5 - y -> dy = -1
        
        g_walls = grad_constraints[:N*4]
        
        # g_walls is indexed [0..N-1] for left, [N..2N-1] for right, etc.
        # Actually it's flattened: [left_0, ..., left_N-1, right_0, ..., right_N-1, ...]
        
        grad_centers[:, 0] += g_walls[0:N]      # Left wall: +x
        grad_centers[:, 0] -= g_walls[N:2*N]    # Right wall: -x
        grad_centers[:, 1] += g_walls[2*N:3*N]  # Bottom wall: +y
        grad_centers[:, 1] -= g_walls[3*N:4*N]  # Top wall: -y
        
        # Pair gradients
        g_pairs = grad_constraints[N*4:]
        
        # For each pair (i, j), constraint is d_ij = ||c_i - c_j|| / 2.
        # Gradient wrt c_i: (c_i - c_j) / (2 * ||c_i - c_j||) * g_pairs[k]
        # Gradient wrt c_j: (c_j - c_i) / (2 * ||c_i - c_j||) * g_pairs[k]
        # Note: dists[idx_i, idx_j] is the full distance.
        # d_ij = dist / 2.
        # grad d_ij / dc_i = (c_i - c_j) / dist * 1/2 ?
        # Let D = ||c_i - c_j||. d = D/2.
        # dd/dc_i = (1/2) * (c_i - c_j) / D.
        
        pair_dists_full = dists[idx_i, idx_j]
        # Avoid division by zero
        mask = pair_dists_full > 1e-12
        g_pairs_safe = np.where(mask, g_pairs, 0.0)
        dists_safe = np.where(mask, pair_dists_full, 1.0)
        
        # Factor = g_pairs / (2 * dists)
        factors = g_pairs_safe / (2.0 * dists_safe)
        
        # Accumulate gradients
        # For each pair k, i = idx_i[k], j = idx_j[k]
        # grad_c_i += (c_i - c_j) / D * factor
        # grad_c_j += (c_j - c_i) / D * factor
        
        # Vectorized accumulation using np.add.at
        diff_vecs = centers[idx_i] - centers[idx_j]
        # diff_vecs shape (M, 2)
        # factors shape (M,)
        # We need to multiply each vector by its factor
        weighted_diffs = diff_vecs * factors[:, np.newaxis]
        
        # Add to i
        np.add.at(grad_centers, idx_i, weighted_diffs)
        # Subtract from j (since grad_c_j = -grad_c_i for this term)
        np.add.at(grad_centers, idx_j, -weighted_diffs)
        
        # Gradient of objective is grad_centers
        # We are minimizing obj = -soft_min.
        # grad(obj) = grad_centers
        
        return obj, grad_centers.flatten()

    def hexagonal_init():
        """Generate a hexagonal lattice initial guess."""
        d = 0.073
        y_spacing = d * math.sqrt(3) / 2
        cols = 15
        rows = 16
        pts = []
        total_height = (rows - 1) * y_spacing
        y_start = -total_height / 2
        for i in range(rows):
            y = y_start + i * y_spacing
            if i % 2 == 0:
                x_start = -0.5 + d/2
                for j in range(cols):
                    x = x_start + j * d
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for j in range(cols - 1):
                    x = x_start + j * d
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            diff = N - len(pts)
            random_pts = np.random.uniform(-0.45, 0.45, (diff, 2))
            pts = np.vstack([pts, random_pts])
        return pts

    def square_init():
        """Generate a square grid initial guess."""
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) > N:
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        return grid

    def random_init():
        return np.random.uniform(-0.45, 0.45, (N, 2))

    best_centers = None
    best_radius = 0.0

    # Multi-restart strategy
    inits = [
        ("hex", hexagonal_init()),
        ("sq", square_init()),
    ]
    
    for name, init_centers in inits:
        if time.time() > deadline - 40:
            break
            
        # Gradient-based optimization
        # Start with a moderate beta, increase it to sharpen the soft-min
        # Beta controls the approximation of the min.
        # Small beta -> smooth, hard to optimize but global.
        # Large beta -> sharp, close to true min, but non-smooth.
        
        current_centers = init_centers.copy()
        flat_params = current_centers.flatten()
        
        # Annealing of beta
        beta_start = 50.0
        beta_end = 200.0
        num_beta_steps = 5
        
        for b_step in range(num_beta_steps):
            if time.time() > deadline - 10:
                break
            beta = beta_start + (beta_end - beta_start) * (b_step / (num_beta_steps - 1))
            
            # Run L-BFGS-B
            # We want to minimize -soft_min
            # Constraints: centers must stay in [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (N * 2)
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad,
                    flat_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                flat_params = res.x
                current_centers = flat_params.reshape(N, 2)
            except Exception:
                pass
            
            # Clip to ensure validity
            current_centers = np.clip(current_centers, -0.5, 0.5)
        
        r = compute_radius(current_centers)
        if r > best_radius:
            best_radius = r
            best_centers = current_centers.copy()

    # If time remains, try more random inits
    num_random_tries = 0
    while time.time() < deadline - 25:
        if time.time() > deadline - 10:
            break
        init_centers = random_init()
        
        # Gradient-based optimization on random init
        flat_params = init_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        # Use a higher beta directly for random inits as they are likely far from optimal
        beta = 100.0
        try:
            res = scipy.optimize.minimize(
                objective_and_grad,
                flat_params,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            flat_params = res.x
            current_centers = flat_params.reshape(N, 2)
        except Exception:
            current_centers = init_centers
        
        current_centers = np.clip(current_centers, -0.5, 0.5)
        r = compute_radius(current_centers)
        if r > best_radius:
            best_radius = r
            best_centers = current_centers.copy()
        
        num_random_tries += 1
        if num_random_tries >= 5:
            break

    # Final polish on best
    if best_centers is not None and time.time() < deadline - 15:
        flat_params = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 2)
        beta = 300.0
        try:
            res = scipy.optimize.minimize(
                objective_and_grad,
                flat_params,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            flat_params = res.x
            final_centers = flat_params.reshape(N, 2)
        except Exception:
            final_centers = best_centers
            
        final_centers = np.clip(final_centers, -0.5, 0.5)
        final_r = compute_radius(final_centers)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final_centers

    # Ensure valid output
    if best_centers is None:
        best_centers = square_init()
        
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation
    final_r = compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to a known valid configuration
        best_centers = square_init()
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
