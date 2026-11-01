"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N_CIRCLES = 200
HALF_SIDE = 0.5


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.08)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def hexagonal_initialization():
    """Create a hexagonal lattice initialization for N circles in unit square."""
    # Hexagonal packing: rows offset
    # Try to fit as many as possible in the square
    # Row spacing: dy = r * sqrt(3), x spacing: dx = 2*r
    # We'll generate a dense hex grid and pick N points closest to center
    
    # Generate a large hex grid
    points = []
    # Use a spacing that gives roughly the right density
    # For 200 circles in unit square, average area per circle = 1/200 = 0.005
    # Hex packing: each circle occupies area = (sqrt(3)/2) * d^2 where d is center-to-center
    # So d^2 = 0.005 * 2/sqrt(3) ≈ 0.00577, d ≈ 0.076
    # Let's use a range of spacings
    
    best_points = None
    best_score = 0.0
    
    for scale in [0.07, 0.072, 0.074, 0.076, 0.078, 0.08]:
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        x_vals = np.arange(-0.5, 0.5 + dx/2, dx)
        y_vals = np.arange(-0.5, 0.5 + dy/2, dy)
        
        grid_points = []
        for i, y in enumerate(y_vals):
            for x in x_vals:
                if i % 2 == 1:
                    x_offset = dx / 2
                else:
                    x_offset = 0
                px = x + x_offset
                if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                    grid_points.append([px, y])
        
        if len(grid_points) < N_CIRCLES:
            continue
        
        grid_points = np.array(grid_points)
        
        # Sort by distance from center and take N closest
        dists = np.sum(grid_points ** 2, axis=1)
        idx = np.argsort(dists)[:N_CIRCLES]
        points = grid_points[idx]
        
        r = compute_radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: simple grid
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]


def objective_and_grad(x, centers_2d, wall_margin=1e-6):
    """
    Objective: minimize negative of min pair distance (to maximize it).
    We use a smooth approximation: minimize -sum of exp(-d_ij^2 / sigma^2) weighted,
    but actually we want to maximize the minimum distance.
    
    Better approach: use a differentiable surrogate.
    We minimize: -r where r = min over all pairs of dist(i,j)/2 and wall distances.
    But min is non-differentiable. Instead, we use:
    
    f(x) = -log(sum(exp(-d_ij^2 / tau)))  -- this is a smooth max-min surrogate
    
    Actually, let's use a simpler approach: 
    Minimize the maximum "violation" or use a potential energy approach.
    
    For L-BFGS-B, let's define:
    f(x) = -min_dist  (we want to maximize min_dist)
    
    Since min is not smooth, we'll use a differentiable approximation:
    f(x) = -(1/tau) * log(sum_{pairs} exp(-tau * d_ij))  approximates -min(d_ij) as tau->inf
    
    But this is expensive. Let's instead use a simpler gradient-based approach:
    At each step, find the closest pair and push them apart.
    
    Actually, let's use a cleaner formulation:
    We'll minimize a smooth objective that penalizes close pairs:
    f(x) = sum_{i<j} exp(-alpha * d_ij^2) + beta * sum_i exp(-alpha * wall_dist_i^2)
    
    This is smooth and its gradient points away from close pairs and walls.
    Minimizing this spreads circles apart.
    """
    alpha = 50.0  # sharpness of the exponential penalty
    beta = 50.0
    
    centers = x.reshape(-1, 2)
    
    # Compute all pairwise distances
    # For 200 points, that's 19900 pairs - manageable
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Zero out diagonal
    np.fill_diagonal(dist_sq, 1e10)
    
    # Only consider pairs that are reasonably close to avoid underflow issues
    # Actually, exp(-alpha * d^2) underflows for large d, which is fine
    
    pair_penalty = np.sum(np.exp(-alpha * dist_sq))
    
    # Wall penalty: distance from each center to nearest wall
    wall_dists = np.minimum(np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
                           np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5))
    wall_penalty = beta * np.sum(np.exp(-alpha * wall_dists ** 2))
    
    f = pair_penalty + wall_penalty
    
    # Gradient
    # d(pair_penalty)/dx_i = sum_j 2*alpha*d_ij^2 * exp(-alpha*d_ij^2) * (x_i - x_j) / d_ij^2
    # Wait, let me recompute:
    # f_pair = sum_{i<j} exp(-alpha * ||x_i - x_j||^2)
    # df/dx_i = sum_{j != i} exp(-alpha * d_ij^2) * (-alpha) * 2 * (x_i - x_j)
    #          = -2*alpha * sum_{j != i} (x_i - x_j) * exp(-alpha * d_ij^2)
    
    # Using the full matrix:
    # grad_i = -2*alpha * sum_j (x_i - x_j) * exp(-alpha * d_ij^2)
    # But we need to be careful with the factor of 2 since we sum over all ordered pairs
    # in the matrix, but the objective sums over unordered pairs.
    # Actually, my dist_sq matrix has all pairs including both (i,j) and (j,i),
    # and I summed all of them, so I've double-counted. Let me fix:
    
    # Recompute properly:
    # f_pair = 0.5 * sum_{i != j} exp(-alpha * d_ij^2)  [to avoid double counting]
    # But I already computed pair_penalty = sum of all entries including both directions
    # So pair_penalty = 2 * (actual sum over i<j)
    # That's fine for optimization, just scales the gradient.
    
    weight_matrix = np.exp(-alpha * dist_sq)
    np.fill_diagonal(weight_matrix, 0)
    
    # grad for pair part: for each i, grad_i = -2*alpha * sum_j weight_ij * (x_i - x_j)
    # = -2*alpha * (x_i * sum_j weight_ij - sum_j weight_ij * x_j)
    
    sum_weights = np.sum(weight_matrix, axis=1)  # shape (N,)
    weighted_sum_x = weight_matrix @ centers  # shape (N, 2)
    
    grad_pair = -2 * alpha * (centers * sum_weights[:, np.newaxis] - weighted_sum_x)
    
    # Wall gradient:
    # f_wall = beta * sum_i exp(-alpha * w_i^2) where w_i = min dist to wall
    # This is tricky because of the min. Let's approximate:
    # For each dimension, the distance to the nearest wall in that direction.
    # w_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
    # The gradient is discontinuous at the kink, but we can use the subgradient
    # pointing away from the nearest wall.
    
    # Simpler: use separate penalties for each wall
    # f_wall = beta * [sum_i exp(-alpha*(0.5-x_i)^2) + sum_i exp(-alpha*(x_i+0.5)^2)
    #               + sum_i exp(-alpha*(0.5-y_i)^2) + sum_i exp(-alpha*(y_i+0.5)^2)]
    
    # This is smoother and differentiable.
    # Let's recompute with this approach:
    x_coords = centers[:, 0]
    y_coords = centers[:, 1]
    
    # Distances to walls
    dx_right = 0.5 - x_coords
    dx_left = x_coords + 0.5
    dy_top = 0.5 - y_coords
    dy_bottom = y_coords + 0.5
    
    # f_wall = beta * sum_i [exp(-alpha*dx_right^2) + exp(-alpha*dx_left^2) 
    #                       + exp(-alpha*dy_top^2) + exp(-alpha*dy_bottom^2)]
    
    # Gradient:
    # d/dx_i of exp(-alpha*dx_right^2) = exp(-alpha*dx_right^2) * (-2*alpha*dx_right) * (-1)
    #                                   = 2*alpha*dx_right*exp(-alpha*dx_right^2)
    # d/dx_i of exp(-alpha*dx_left^2) = exp(-alpha*dx_left^2) * (-2*alpha*dx_left) * (1)
    #                                  = -2*alpha*dx_left*exp(-alpha*dx_left^2)
    
    grad_wall_x = beta * (2 * alpha * dx_right * np.exp(-alpha * dx_right ** 2) 
                         - 2 * alpha * dx_left * np.exp(-alpha * dx_left ** 2))
    grad_wall_y = beta * (2 * alpha * dy_top * np.exp(-alpha * dy_top ** 2) 
                         - 2 * alpha * dy_bottom * np.exp(-alpha * dy_bottom ** 2))
    
    grad = np.zeros_like(x)
    grad[0::2] = grad_pair[:, 0] + grad_wall_x
    grad[1::2] = grad_pair[:, 1] + grad_wall_y
    
    return f, grad


def gradient_optimize(centers, max_iters=50, alpha_start=20.0, alpha_end=100.0):
    """Run L-BFGS-B optimization with adaptive alpha."""
    x0 = centers.flatten()
    best = centers.copy()
    best_r = compute_radius(best)
    
    n_alphas = 5
    alphas = np.linspace(alpha_start, alpha_end, n_alphas)
    
    for alpha in alphas:
        # Define objective with current alpha
        def obj(x):
            return objective_and_grad(x, None, alpha)[0]
        
        def grad(x):
            return objective_and_grad(x, None, alpha)[1]
        
        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                             bounds=[(-0.5, 0.5)] * (2 * N_CIRCLES),
                             options={'maxiter': max_iters, 'gtol': 1e-6, 'maxls': 50})
            new_centers = result.x.reshape(-1, 2)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            new_r = compute_radius(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best = new_centers.copy()
            x0 = new_centers
        except Exception:
            pass
    
    return best, best_r


def push_apart(centers, n_rounds=30, r_target=0.037):
    """Iteratively push overlapping circles apart."""
    current = centers.copy()
    
    for _ in range(n_rounds):
        tree = cKDTree(current)
        pairs = tree.query_pairs(r=2 * r_target)
        
        if len(pairs) == 0:
            break
        
        moved_any = False
        for i, j in pairs:
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random nudge
                current[i] += np.random.normal(0, 0.001, 2)
                current[j] -= np.random.normal(0, 0.001, 2)
                moved_any = True
                continue
            
            min_dist = 2 * r_target
            if dist < min_dist:
                push = (min_dist - dist) / 2 + 0.0005
                diff_norm = diff / dist
                current[i] += push * diff_norm
                current[j] -= push * diff_norm
                moved_any = True
        
        # Clamp to boundary
        current = np.clip(current, -0.5 + 1e-9, 0.5 - 1e-9)
        
        if not moved_any:
            break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    # Step 1: Hexagonal initialization
    best_config = hexagonal_initialization()
    best_r = compute_radius(best_config)
    
    # Step 2: Initial push-apart to resolve obvious overlaps
    if time.time() < deadline - 10:
        pushed = push_apart(best_config, n_rounds=50, r_target=0.036)
        pushed_r = compute_radius(pushed)
        if pushed_r > best_r:
            best_r = pushed_r
            best_config = pushed
    
    # Step 3: Gradient-based optimization with multiple alpha levels
    if time.time() < deadline - 5:
        opt_result, opt_r = gradient_optimize(best_config, max_iters=30)
        if opt_r > best_r:
            best_r = opt_r
            best_config = opt_result
    
    # Step 4: Another push-apart round
    if time.time() < deadline - 5:
        pushed = push_apart(best_config, n_rounds=30, r_target=best_r * 2)
        pushed_r = compute_radius(pushed)
        if pushed_r > best_r:
            best_r = pushed_r
            best_config = pushed
    
    # Step 5: More gradient refinement
    if time.time() < deadline - 3:
        opt_result, opt_r = gradient_optimize(best_config, max_iters=20)
        if opt_r > best_r:
            best_r = opt_r
            best_config = opt_result
    
    # Step 6: Final push-apart
    if time.time() < deadline - 2:
        pushed = push_apart(best_config, n_rounds=20, r_target=best_r * 2)
        pushed_r = compute_radius(pushed)
        if pushed_r > best_r:
            best_r = pushed_r
            best_config = pushed
    
    # Final check
    final_r = compute_radius(best_config)
    if final_r <= 0:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
    
    return best_config
# EVOLVE-BLOCK-END
