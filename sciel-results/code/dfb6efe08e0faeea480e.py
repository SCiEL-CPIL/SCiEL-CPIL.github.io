"""Improved packing program using gradient-based optimization with soft-min surrogate."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_boundary)

def _soft_min_logsumexp(x, beta=100.0):
    """Compute soft-min of array x using log-sum-exp trick."""
    # soft-min(x) = -soft-max(-x)
    # soft-max(x) = (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, use max subtraction
    x = -x  # negative for soft-min
    x_max = np.max(x)
    if x_max > 1000:
        x_max = 1000
    shifted = beta * (x - x_max)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max_val = (1.0 / beta) * log_sum + x_max
    return -soft_max_val  # negative because we negated

def _soft_objective_and_grad(pos, beta=100.0):
    """
    Compute a smooth surrogate for the negative radius and its gradient.
    We want to maximize r = min(min_pair/2, min_boundary).
    So we minimize -r.
    
    Using soft-min: soft_min(x) approximates min(x) for large beta.
    We define:
      d_boundary = min_i min(0.5 - |x_i|, 0.5 - |y_i|)
      d_pair = min_{i<j} |c_i - c_j| / 2
      r = min(d_boundary, d_pair)
    
    We use soft-min to smooth both.
    """
    n = len(pos)
    if n == 0:
        return 0.0, np.array([])
    
    # Boundary distances
    # For each point, distance to each of 4 walls
    # Left wall: 0.5 + x_i (since x >= -0.5, dist = x_i - (-0.5) = x_i + 0.5)
    # Right wall: 0.5 - x_i
    # Bottom wall: 0.5 + y_i
    # Top wall: 0.5 - y_i
    d_left = 0.5 + pos[:, 0]
    d_right = 0.5 - pos[:, 0]
    d_bottom = 0.5 + pos[:, 1]
    d_top = 0.5 - pos[:, 1]
    
    # Clip to be non-negative for safety
    d_left = np.maximum(d_left, 0)
    d_right = np.maximum(d_right, 0)
    d_bottom = np.maximum(d_bottom, 0)
    d_top = np.maximum(d_top, 0)
    
    # Stack all boundary distances
    boundary_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Pairwise distances / 2
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dist_sq[iu]) / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pair_dists])
    
    # Soft-min
    beta_val = beta
    x = all_dists
    x_max = np.max(x)
    if x_max > 1000:
        x_max = 1000
    shifted = beta_val * (x - x_max)
    shifted = np.clip(shifted, -700, 700)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        sum_exp = 1e-300
    log_sum = np.log(sum_exp)
    soft_min_val = (1.0 / beta_val) * log_sum + x_max
    
    # Now compute gradient with respect to pos
    # d(soft_min)/d(x_k) = -exp(beta*(x_k - x_max)) / sum_exp
    # (negative because soft_min = -soft_max(-x), and d(soft_max)/d(x_k) = exp(...)/sum)
    # Actually: soft_min(x) = -(1/beta) log(sum(exp(beta * x))) + x_max
    # d(soft_min)/dx_k = -(1/beta) * (1/sum) * exp(beta*x_k) * beta = -exp(beta*x_k)/sum
    # But we have the shifted version:
    # soft_min = (1/beta)*log(sum(exp(beta*(x_k - x_max)))) + x_max
    # d(soft_min)/dx_k = (1/beta) * (1/sum) * exp(beta*(x_k-x_max)) * beta = exp(beta*(x_k-x_max))/sum
    # Wait, let me re-derive.
    
    # Let f(x) = (1/beta) * log(sum_j exp(beta*(x_j - x_max))) + x_max
    # df/dx_k = (1/beta) * (1/sum) * exp(beta*(x_k-x_max)) * beta = exp(beta*(x_k-x_max)) / sum
    # So the weight for each distance is w_k = exp(beta*(x_k - x_max)) / sum_exp
    
    weights = exp_vals / sum_exp
    
    # Now, we need d(soft_min)/d(pos)
    # soft_min depends on all_dists which depends on pos
    # d(soft_min)/d(pos) = sum_k weights[k] * d(all_dists[k])/d(pos)
    
    grad = np.zeros_like(pos)
    
    # Boundary distances gradients
    # d_left = 0.5 + x_i, d/dx_i = 1, d/dy_i = 0
    # d_right = 0.5 - x_i, d/dx_i = -1
    # d_bottom = 0.5 + y_i, d/dy_i = 1
    # d_top = 0.5 - y_i, d/dy_i = -1
    
    # weights[0:n] correspond to d_left
    # weights[n:2n] correspond to d_right
    # weights[2n:3n] correspond to d_bottom
    # weights[3n:4n] correspond to d_top
    
    w_left = weights[0:n]
    w_right = weights[n:2*n]
    w_bottom = weights[2*n:3*n]
    w_top = weights[3*n:4*n]
    
    grad[:, 0] += w_left - w_right
    grad[:, 1] += w_bottom - w_top
    
    # Pairwise distances
    # pair_dists[m] = |c_{i_m} - c_{j_m}| / 2 where (i_m, j_m) = iu[m]
    # d(pair_dists[m])/d(c_{i_m}) = (c_{i_m} - c_{j_m}) / (2 * |c_{i_m} - c_{j_m}|) = (c_i - c_j) / (2*dist)
    # d(pair_dists[m])/d(c_{j_m}) = (c_{j_m} - c_{i_m}) / (2 * |c_{i_m} - c_{j_m}|) = (c_j - c_i) / (2*dist)
    
    pair_weights = weights[4*n:]
    i_idx = iu[0]
    j_idx = iu[1]
    
    # Compute unit vectors
    # diff[i,j] = c_i - c_j
    # For pair (i, j): dist = |c_i - c_j|
    # d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist)
    # d(dist/2)/d(c_j) = (c_j - c_i) / (2 * dist) = -(c_i - c_j) / (2 * dist)
    
    # We need to accumulate for each point
    # For each pair m with weight w_m:
    #   grad[i_m] += w_m * (c_i - c_j) / (2 * dist_m)
    #   grad[j_m] -= w_m * (c_i - c_j) / (2 * dist_m)
    
    # dist_m = |c_i - c_j|, so (c_i - c_j) / (2*dist_m) is the unit vector from j to i, divided by 2
    # Actually: (c_i - c_j) / dist_m is unit vector from j to i
    # So (c_i - c_j) / (2*dist_m) = 0.5 * unit_vec
    
    # Let's compute efficiently
    # diff_ij = pos[i] - pos[j] for each pair
    diff_pairs = pos[i_idx] - pos[j_idx]
    dist_pairs = np.sqrt(np.sum(diff_pairs**2, axis=1))
    dist_pairs = np.maximum(dist_pairs, 1e-12)
    
    # unit_vec = diff_pairs / dist_pairs[:, np.newaxis]
    # contribution to grad[i] = w_m * 0.5 * unit_vec
    # contribution to grad[j] = -w_m * 0.5 * unit_vec
    
    contrib = 0.5 * pair_weights[:, np.newaxis] * diff_pairs / dist_pairs[:, np.newaxis]
    
    # Accumulate
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    # The objective is -soft_min, so we minimize -soft_min
    # d(-soft_min)/d(pos) = -grad(soft_min)/d(pos) = -grad
    # So the gradient of the objective (which we want to minimize) is -grad
    # For L-BFGS-B, we pass the gradient of the function to minimize
    objective = -soft_min_val
    grad_obj = -grad
    
    return objective, grad_obj

def _initial_hex_grid(n, spacing=None):
    """Generate a hexagonal grid initialization."""
    if spacing is None:
        # Estimate spacing based on density
        # Area per circle ~ (2*r)^2 * sqrt(3)/2 for hex packing
        # r ~ 0.0366, so 2*r ~ 0.073
        # density = n / area = 200 / 1 = 200
        # spacing ~ 1/sqrt(density) ~ 0.07
        spacing = 1.0 / math.sqrt(n) * 0.95
    
    points = []
    dy = spacing * math.sqrt(3) / 2.0
    max_coord = 0.49
    
    n_rows = int(np.ceil(2 * max_coord / dy)) + 2
    n_cols = int(np.ceil(2 * max_coord / spacing)) + 2
    
    for row in range(n_rows):
        y = -max_coord + row * dy
        if y > max_coord:
            break
        x_offset = 0.0 if row % 2 == 0 else spacing / 2.0
        for col in range(n_cols):
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, subsample
    if len(points) > n:
        # Use a simple random subsample
        idx = np.random.RandomState(42).choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        rng = np.random.RandomState(42)
        extra = rng.uniform(-max_coord, max_coord, size=(n - len(points), 2))
        points = np.vstack([points, extra])
    
    return points

def _initial_random(n, seed=42):
    """Generate a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.49, 0.49, size=(n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # Try multiple initializations
    initializations = []
    
    # Hex grid with different spacings
    for spacing in [0.06, 0.065, 0.07, 0.075]:
        initializations.append(_initial_hex_grid(N, spacing=spacing))
    
    # Random initializations
    for seed in [42, 123, 456]:
        initializations.append(_initial_random(N, seed=seed))
    
    # For each initialization, run gradient-based optimization
    for init_idx, pos in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        try:
            # Use scipy's L-BFGS-B
            from scipy.optimize import minimize
            
            # Define the objective function
            def obj_and_grad(x):
                pos_2d = x.reshape(-1, 2)
                # Clip to valid range
                pos_2d = np.clip(pos_2d, -0.4999, 0.4999)
                val, grad = _soft_objective_and_grad(pos_2d, beta=50.0)
                return val, grad.reshape(-1)
            
            # Run optimization
            result = minimize(
                obj_and_grad,
                pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_pos = result.x.reshape(-1, 2)
                opt_pos = np.clip(opt_pos, -0.4999, 0.4999)
                
                r = _compute_radius(opt_pos)
                
                if r > best_r:
                    best_r = r
                    best_pos = opt_pos.copy()
                    
        except Exception as e:
            continue
    
    # If we have a valid solution, try a final polish with higher beta
    if best_pos is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            def obj_and_grad_polish(x):
                pos_2d = x.reshape(-1, 2)
                pos_2d = np.clip(pos_2d, -0.4999, 0.4999)
                val, grad = _soft_objective_and_grad(pos_2d, beta=200.0)
                return val, grad.reshape(-1)
            
            result = minimize(
                obj_and_grad_polish,
                best_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 0:
                opt_pos = result.x.reshape(-1, 2)
                opt_pos = np.clip(opt_pos, -0.4999, 0.4999)
                
                r = _compute_radius(opt_pos)
                
                if r > best_r:
                    best_r = r
                    best_pos = opt_pos.copy()
                    
        except Exception:
            pass
    
    # Fallback: if no valid solution, use a simple grid
    if best_pos is None or best_r < 0.01:
        side = int(np.ceil(math.sqrt(N)))
        best_pos = np.zeros((N, 2))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_pos[count, 0] = -0.45 + 0.9 * i / (side - 1) if side > 1 else 0
                best_pos[count, 1] = -0.45 + 0.9 * j / (side - 1) if side > 1 else 0
                count += 1
            if count >= N:
                break
    
    return best_pos
# EVOLVE-BLOCK-END
