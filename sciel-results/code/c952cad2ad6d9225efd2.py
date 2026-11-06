"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # 1. Initialisation: hexagonal lattice
    # For 200 circles in a unit square, an 8x25 or 10x20 grid is close.
    # Let's use a hexagonal lattice for better initial density.
    # Approximate rows: ~15 rows, ~13-14 cols.
    # Let's just use a square grid first, then optimize.
    # Actually, a hexagonal packing is denser.
    # Let's generate points on a hexagonal grid clipped to the square.
    
    # Grid parameters for hex packing
    # r is the radius, d = 2r is the distance.
    # Let's start with a guess for r. Best known is ~0.0366.
    # d = 0.0732
    # Let's use a slightly smaller d to start, say 0.07.
    
    # Alternative: Start with a square grid of side 1, N=200.
    # m = ceil(sqrt(200)) = 15. 15x15=225.
    # We can remove some to get 200, or just use 200 points from a 15x15 grid.
    
    m = 15
    x = (np.arange(m) + 0.5) / m - 0.5
    y = (np.arange(m) + 0.5) / m - 0.5
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    
    # Shuffle to break symmetry? No, keep structured for now.
    # If we have 225 points, we can drop 25 randomly or from edges.
    # Let's just take the first 200.
    pts = pts[:N]
    
    # 2. Optimization: Gradient Descent with soft constraints
    # Objective: Maximize min( min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|) )
    # Let r be the radius. We want to maximize r.
    # Equivalent to: Minimize -r, subject to:
    #   |c_i - c_j| >= 2r  for all i<j
    #   1/2 - |c_ik| >= r  for all i,k
    
    # We can use a soft penalty approach.
    # Energy E = sum_{i<j} max(0, 2r - |c_i - c_j|)^2 + sum_{i,k} max(0, r - (1/2 - |c_ik|))^2
    # But r is also a variable we want to maximize.
    # Alternatively, fix r and check validity, or use a Lagrangian.
    # Better: Treat r as a variable and use a penalty method.
    # Minimize: P(r, pts) = -r + lambda * (sum violations^2)
    # Or maximize: r - lambda * (sum violations^2)
    
    # Let's use a simple gradient ascent on r and gradient descent on violations.
    # Actually, it's easier to define an objective function that we want to MINIMIZE:
    # J = sum_{i<j} (max(0, 2r - d_ij))^2 + sum_{i,k} (max(0, r - dist_to_wall))^2 - alpha * r
    # We want to find a configuration where the first two terms are 0, and r is large.
    
    # Let's iterate:
    # 1. Start with small r.
    # 2. Increase r slowly.
    # 3. For each r, minimize the penalty term by moving points.
    
    r_current = 0.01
    r_target = 0.036612798904
    r_step = 0.0001
    
    lr = 0.01  # Learning rate for point positions
    lr_r = 0.001 # Learning rate for r? Or just increase r manually.
    
    # Let's use a more robust approach:
    # Minimize F(pts) = sum_{i<j} (max(0, 2r - d_ij))^2 + sum_{i,k} (max(0, r - dist_to_wall))^2
    # And try to increase r as much as possible such that F(pts) < epsilon.
    
    # Actually, let's just use gradient descent on the negative of the minimum distance.
    # This is non-differentiable.
    # Let's use the soft-min formulation.
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # We want to maximize soft_min( min_{i<j} d_ij/2, min_{i,k} (1/2 - |c_ik|) )
    
    beta = 50.0  # Sharpness of soft-min
    
    def compute_soft_min(pts):
        # Pairwise distances
        # d_ij = ||c_i - c_j||
        # d_ik = 1/2 - |c_ik| (distance to wall)
        
        # Vectorized pairwise distance calculation is O(N^2). N=200 -> 40000 pairs.
        # This is fast enough.
        
        # 1. Pairwise distances
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, np.inf)
        
        # More memory efficient:
        # dists = cdist(pts, pts)
        
        # 2. Wall distances
        # dist_to_wall = 0.5 - np.abs(pts)
        # min_dist_to_wall = np.min(dist_to_wall, axis=1)
        
        # Combine all distances (d_ij/2 and d_ik)
        # We need the min of all these values.
        
        # Let's compute the values explicitly.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0  # Radius constraint: d_ij >= 2r -> d_ij/2 >= r
        
        # Wall distances
        # For each point, distance to nearest wall is min(0.5 - x, 0.5 - y, 0.5 + x, 0.5 + y)
        # Since pts are in [-0.5, 0.5], 0.5 - |x| is the distance to the vertical walls.
        # Wait, if x is in [-0.5, 0.5], distance to x=0.5 is 0.5-x, to x=-0.5 is x+0.5.
        # Min is 0.5 - |x|.
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist_per_point = np.min(wall_dists, axis=1)
        
        # Flatten all distances
        all_dists = np.concatenate([pair_dists.ravel(), min_wall_dist_per_point])
        
        # Soft min
        # To avoid overflow, subtract max
        m = np.max(all_dists)
        soft_min_val = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_dists - m))))
        
        # Gradient of soft_min wrt pts
        # d(soft_min)/d(all_dists_k) = (1/sum(exp)) * exp(-beta*(all_dists_k - m))
        # Let w_k = exp(-beta*(all_dists_k - m)) / sum(exp(-beta*(all_dists_j - m)))
        # Then grad_soft_min = -w_k (since soft_min = -1/beta log(sum exp(-beta d)) + m? No.)
        
        # Let S = sum(exp(-beta * d_j))
        # soft_min = -1/beta * log(S)
        # d(soft_min)/d(d_k) = -1/beta * (1/S) * (-beta * exp(-beta * d_k)) = exp(-beta * d_k) / S
        
        # So the weight for each distance is proportional to exp(-beta * d_k).
        # The distances with smaller d_k have larger weights.
        
        exp_vals = np.exp(-beta * (all_dists - m))
        weights = exp_vals / np.sum(exp_vals)
        
        # Now, we need to map these weights back to the gradients wrt pts.
        # grad = sum_k weights_k * d(all_dists_k)/d(pts)
        
        # all_dists is constructed from pair_dists and min_wall_dist.
        # This is tricky to vectorize the gradient directly without backprop.
        
        # Let's compute gradients for pair distances and wall distances separately.
        
        # 1. Wall distances gradient
        # min_wall_dist_per_point is the min of (0.5-|x|, 0.5-|y|).
        # The gradient is non-zero only for the coordinate that achieves the min.
        # If 0.5-|x| < 0.5-|y|, then d/dx = -sign(x), d/dy = 0.
        # If 0.5-|y| < 0.5-|x|, then d/dy = -sign(y), d/dx = 0.
        # If equal, split the weight.
        
        # Let's compute the weights for wall distances.
        # The indices for wall distances are from N*N to N*N + N - 1.
        wall_weights = weights[N*N:]
        
        # Gradient from walls
        grad_wall = np.zeros_like(pts)
        # For each point i:
        # d_i = min(0.5-|x_i|, 0.5-|y_i|)
        # If 0.5-|x_i| < 0.5-|y_i|:
        #   grad_x += wall_weights[i] * (-sign(x_i))
        #   grad_y += 0
        # Else if 0.5-|y_i| < 0.5-|x_i|:
        #   grad_x += 0
        #   grad_y += wall_weights[i] * (-sign(y_i))
        # Else:
        #   grad_x += wall_weights[i] * 0.5 * (-sign(x_i))
        #   grad_y += wall_weights[i] * 0.5 * (-sign(y_i))
        
        abs_x = np.abs(pts[:, 0])
        abs_y = np.abs(pts[:, 1])
        dist_x = 0.5 - abs_x
        dist_y = 0.5 - abs_y
        
        mask_x = dist_x < dist_y
        mask_y = dist_y < dist_x
        mask_eq = (dist_x == dist_y)
        
        # sign(x) is 1 if x>0, -1 if x<0, 0 if x=0.
        sign_x = np.sign(pts[:, 0])
        sign_y = np.sign(pts[:, 1])
        
        grad_wall[:, 0] += np.where(mask_x, wall_weights, 0) * (-sign_x)
        grad_wall[:, 1] += np.where(mask_y, wall_weights, 0) * (-sign_y)
        
        # Handle equality
        # If equal, split weight 50/50
        grad_wall[:, 0] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_x)
        grad_wall[:, 1] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_y)
        
        # 2. Pairwise distances gradient
        # pair_dists = dists / 2
        # dists = sqrt(sum(diff^2))
        # d(pair_dists_ij)/d(c_i) = -0.5 * (c_i - c_j) / dists_ij
        # d(pair_dists_ij)/d(c_j) = 0.5 * (c_i - c_j) / dists_ij
        
        # We have weights for all pairs.
        pair_weights = weights[:N*N].reshape(N, N)
        
        # Gradient for point i from all pairs (i, j)
        # grad_i = sum_j pair_weights[i, j] * d(pair_dists_ij)/d(c_i)
        #        = sum_j pair_weights[i, j] * (-0.5 * (c_i - c_j) / dists_ij)
        
        # Note: pair_weights is symmetric? No, weights are based on distances, which are symmetric.
        # But the sum should be taken over all j != i.
        
        # Let's compute the vector field.
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # coeff = -0.5 * pair_weights / dists
        # coeff = np.where(np.eye(N, dtype=bool), 0, coeff)
        
        # grad_pair_i = np.sum(coeff * diff, axis=1)
        
        # Let's do it carefully.
        # diff[i, j] = c_i - c_j
        # d(pair_dists_ij)/d(c_i) = -0.5 * diff[i, j] / dists[i, j]
        # So term for c_i is -0.5 * pair_weights[i, j] * diff[i, j] / dists[i, j]
        
        # Sum over j for each i.
        
        # We can use matrix multiplication?
        # coeff[i, j] = -0.5 * pair_weights[i, j] / dists[i, j]
        # grad_pair[i] = sum_j coeff[i, j] * diff[i, j]
        
        # diff is (N, N, 2). coeff is (N, N).
        # We can do: np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Let's compute dists again (we have it from before, but let's be safe).
        # We already computed diff and dists above.
        
        # Wait, I computed diff and dists inside the function? No, I computed them in the main loop?
        # No, I need to compute them inside the function or pass them.
        # Let's restructure to compute gradients efficiently.
        
        # This function is getting complex. Let's write a helper.
        
        return soft_min_val, grad_wall # Placeholder, need to add pair grad
    
    # Let's rewrite the gradient calculation properly.
    
    def compute_objective_and_grad(pts):
        # 1. Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # 2. Wall distances
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist_per_point = np.min(wall_dists, axis=1)
        
        # 3. Combine
        all_dists = np.concatenate([pair_dists.ravel(), min_wall_dist_per_point])
        
        # 4. Soft min
        m = np.max(all_dists)
        exp_vals = np.exp(-beta * (all_dists - m))
        sum_exp = np.sum(exp_vals)
        soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
        weights = exp_vals / sum_exp
        
        # 5. Gradients
        grad = np.zeros_like(pts)
        
        # Wall gradients
        wall_weights = weights[N*N:]
        abs_x = np.abs(pts[:, 0])
        abs_y = np.abs(pts[:, 1])
        dist_x = 0.5 - abs_x
        dist_y = 0.5 - abs_y
        
        mask_x = dist_x < dist_y
        mask_y = dist_y < dist_x
        mask_eq = (dist_x == dist_y)
        
        sign_x = np.sign(pts[:, 0])
        sign_y = np.sign(pts[:, 1])
        
        grad[:, 0] += np.where(mask_x, wall_weights, 0) * (-sign_x)
        grad[:, 1] += np.where(mask_y, wall_weights, 0) * (-sign_y)
        
        grad[:, 0] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_x)
        grad[:, 1] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_y)
        
        # Pair gradients
        pair_weights = weights[:N*N].reshape(N, N)
        
        # coeff[i, j] = -0.5 * pair_weights[i, j] / dists[i, j]
        # Avoid division by zero (diagonal is inf, so 1/inf = 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff = -0.5 * pair_weights / dists
        coeff = np.where(np.eye(N, dtype=bool), 0, coeff)
        
        # grad_pair[i] = sum_j coeff[i, j] * diff[i, j]
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        grad += grad_pair
        
        # We want to MAXIMIZE soft_min, so we move in the direction of the gradient.
        # pts += lr * grad
        
        return soft_min_val, grad
    
    # Optimization Loop
    # We want to maximize soft_min.
    # Use gradient ascent.
    
    # Adaptive learning rate?
    # Start with small lr.
    
    lr = 0.005
    best_val = -np.inf
    best_pts = pts.copy()
    
    # Momentum?
    v = np.zeros_like(pts)
    mu = 0.9
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline and steps < max_steps:
        val, grad = compute_objective_and_grad(pts)
        
        # Update
        v = mu * v + (1 - mu) * grad
        pts = pts + lr * v
        
        # Keep points inside the square?
        # The soft constraint should handle it, but let's clip to be safe?
        # Clipping might break gradients. Let's rely on the penalty.
        # However, if points go far outside, distances to walls become negative,
        # soft_min will be very negative, and gradient will push them back.
        
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
        
        # Decay learning rate?
        # lr = lr * 0.999
        
        steps += 1
        
        # Check convergence?
        # If grad is small, break.
        # if np.max(np.abs(grad)) < 1e-6:
        #     break
        
    # Final check
    # Compute the true minimum radius for best_pts
    def compute_true_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)
    
    r_final = compute_true_radius(best_pts)
    
    # If the result is invalid (e.g., points outside), return the initial grid?
    # The problem says: "A centre on or outside the square... makes the candidate invalid."
    # So we must ensure all points are strictly inside (or on boundary? "inside" usually means closed set, but "on or outside" invalid implies strict interior? No, "on or outside" -> invalid. So must be strictly inside? Or "on" is invalid? "on or outside" -> invalid. So must be strictly inside.
    # Let's check the constraint: min_i min_k (1/2 - |c_ik|) > 0.
    
    if r_final <= 0:
        # Fallback to initial grid
        return pts
    
    return best_pts

# EVOLVE-BLOCK-END
