"""
Strategy: Deterministic Gradient-Based Max-Min Optimization with Targeted Perturbations.
Instead of random walks, we use a smooth surrogate for the minimum distance (soft-min)
and optimize it using L-BFGS-B. This allows for precise local improvements.
We periodically "kick" the points involved in the closest pairs to escape local minima,
then re-optimize. This combines the precision of gradient methods with the
exploration capability of targeted restarts.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
def _soft_min_score(centers, sharpness=50.0):
    """
    Computes a smooth approximation of the minimum radius.
    Uses a soft-min on the squared distances to avoid non-differentiability.
    Returns a scalar value to be maximized (so we minimize the negative).
    """
    # Distance to boundary
    # Square is [-0.5, 0.5]
    # dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # We want to maximize the minimum of:
    # 1. min_{i<j} |c_i - c_j| / 2
    # 2. min_i min_k (0.5 - |c_ik|)
    
    # To make it differentiable and efficient, we work with squared distances.
    # Let d_ij = |c_i - c_j|^2.
    # Let b_i = min((0.5-|x_i|)^2, (0.5-|y_i|)^2).
    
    # We want to maximize min( sqrt(min d_ij)/2, min b_i )
    # Equivalent to maximizing min( min d_ij / 4, min b_i )
    # Let's just maximize the soft-min of the set of all relevant squared distances.
    # Relevant distances:
    # - Pairwise squared distances
    # - Boundary squared distances
    
    # For N=200, pairwise is ~20k. Boundary is 200. Total ~20.2k.
    # Computing soft-min on 20k elements every iteration might be slow if done naively.
    # But L-BFGS-B needs gradients.
    
    # Let's construct the vector of squared distances.
    # x = centers[:, 0], y = centers[:, 1]
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Pairwise squared distances
    # Use broadcasting for N=200 (200x200 matrix is fine)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq_pairs = dx**2 + dy**2
    
    # Remove diagonal (self-distance)
    # We need to flatten and exclude diagonal indices
    # Mask for upper triangle to avoid duplicates and diagonal
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist_sq_flat = dist_sq_pairs[mask]
    
    # Boundary squared distances
    # For each point, dist to nearest wall squared.
    # d_x = 0.5 - |x|, d_y = 0.5 - |y|
    # d_min = min(d_x, d_y)
    # We want to maximize min(d_min).
    # Let's include d_x^2 and d_y^2 in the pool?
    # If we include both, the soft-min will be pulled down by the smaller one, which is correct.
    # But wait, if a point is close to left wall (x=-0.5), d_x is small. d_y might be large.
    # min(d_x, d_y) is d_x.
    # If we put d_x^2 and d_y^2 in the list, the min is d_x^2.
    # So yes, include both.
    
    d_x = 0.5 - np.abs(x)
    d_y = 0.5 - np.abs(y)
    
    # Ensure non-negative (should be if inside, but during optimization might go out)
    # If it goes out, d becomes negative, squared is positive.
    # This is a problem: if x = 0.6, d_x = -0.1, d_x^2 = 0.01.
    # But we want to penalize being outside.
    # Actually, the task says "inside the square".
    # If we allow optimization to go outside, the score becomes 0 in the real metric.
    # In the soft-min metric, it might look "okay" if it goes far out? No, 0.01 is small.
    # But if it goes to x=1.0, d_x = -0.5, d_x^2 = 0.25. That's worse than 0.01.
    # However, the real constraint is hard.
    # To be safe, we should penalize violations heavily or constrain the optimizer.
    # L-BFGS-B supports bounds. Let's use bounds [-0.5, 0.5] for all coordinates.
    # This ensures we never evaluate outside.
    
    # So d_x and d_y are always >= 0.
    # We add d_x^2 and d_y^2 to the distance pool.
    
    dist_sq_boundary = np.concatenate([d_x**2, d_y**2])
    
    # Combine all squared distances
    all_dist_sq = np.concatenate([dist_sq_flat, dist_sq_boundary])
    
    # Soft min: -1/sharpness * log(sum(exp(-sharpness * d)))
    # This approximates min(d) as sharpness -> infinity.
    # We want to MAXIMIZE the min distance.
    # So we MINIMIZE -soft_min.
    # Or we can just return -soft_min and let minimize handle it.
    
    # To avoid overflow in exp, use log-sum-exp trick.
    # Let a_i = -sharpness * d_i
    # sum(exp(a_i)) = exp(max(a)) * sum(exp(a_i - max(a)))
    # log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
    # soft_min = -1/sharpness * (max(a) + log(sum(exp(a_i - max(a)))))
    
    a = -sharpness * all_dist_sq
    a_max = np.max(a)
    # If a_max is very large positive (d is very negative? No, d is squared dist, so >=0)
    # d >= 0, so a <= 0.
    # a_max is the least negative, i.e., corresponding to smallest d.
    
    # exp(a - a_max) is <= 1. Safe.
    log_sum_exp = a_max + np.log(np.sum(np.exp(a - a_max)))
    
    soft_min_val = -1.0 / sharpness * log_sum_exp
    
    # We want to maximize soft_min_val.
    # So objective function for minimize is -soft_min_val.
    return -soft_min_val

def _objective_and_grad(centers_flat, sharpness=50.0):
    """
    Computes objective and gradient for L-BFGS-B.
    centers_flat: shape (2N,)
    """
    centers = centers_flat.reshape(N, 2)
    
    # Calculate Objective
    obj = _soft_min_score(centers, sharpness)
    
    # Calculate Gradient
    # We need d(obj)/d(centers)
    # obj = -soft_min
    # soft_min = -1/s * L, where L = log(sum(exp(-s * d_i)))
    # d(soft_min)/d(c_k) = -1/s * dL/dc_k
    # dL/dc_k = (1/sum(exp(-s*d_i))) * sum_i (d(exp(-s*d_i))/dc_k)
    # d(exp(-s*d_i))/dc_k = exp(-s*d_i) * (-s * d(d_i)/dc_k)
    
    # This requires storing the exp terms. For N=200, 20k terms.
    # Storing 20k floats is fine.
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq_pairs = dx**2 + dy**2
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # We need to map back from flat indices to (i,j) for gradient accumulation.
    # This is tricky. 
    # Alternative: Compute gradient of the sum of exp terms directly on the matrix.
    
    # Let's reconstruct the full matrix of exp terms for pairwise.
    # d_ij = (x_i - x_j)^2 + (y_i - y_j)^2
    # term_ij = exp(-s * d_ij)
    
    # We only care about i < j.
    # Let's compute the full matrix of terms, zero out diagonal and lower triangle.
    
    a_pairs = -sharpness * dist_sq_pairs
    # Zero out diagonal and lower triangle
    a_pairs = np.where(mask, a_pairs, 0.0)
    
    # Boundary terms
    d_x = 0.5 - np.abs(x)
    d_y = 0.5 - np.abs(y)
    # d_x^2 and d_y^2
    # term_bx_i = exp(-s * d_x_i^2)
    # term_by_i = exp(-s * d_y_i^2)
    
    a_bx = -sharpness * (d_x**2)
    a_by = -sharpness * (d_y**2)
    
    # Total sum of exps
    # S = sum(a_pairs_exp) + sum(exp(a_bx)) + sum(exp(a_by))
    # To avoid overflow/underflow, we used log-sum-exp for value.
    # For gradient, we can use the weights w_i = exp(a_i - a_max) / sum(exp(a_i - a_max))
    # Then dL/dc_k = sum_i w_i * (-s * d(d_i)/dc_k)
    
    # Let's compute weights.
    # We need to handle the max.
    # a_max_global = max(max(a_pairs), max(a_bx), max(a_by))
    
    a_max_pairs = np.max(a_pairs) if np.any(mask) else -np.inf
    a_max_bx = np.max(a_bx)
    a_max_by = np.max(a_by)
    
    a_max = max(a_max_pairs, a_max_bx, a_max_by)
    
    # If a_max is -inf (no terms?), handle gracefully.
    if not np.isfinite(a_max):
        return obj, np.zeros_like(centers_flat)
        
    # Compute weights relative to a_max
    # w_pairs = exp(a_pairs - a_max)
    # But we only want upper triangle.
    w_pairs = np.exp(a_pairs - a_max)
    w_pairs = np.where(mask, w_pairs, 0.0)
    
    w_bx = np.exp(a_bx - a_max)
    w_by = np.exp(a_by - a_max)
    
    sum_w = np.sum(w_pairs) + np.sum(w_bx) + np.sum(w_by)
    
    if sum_w == 0:
        return obj, np.zeros_like(centers_flat)
    
    # Normalize weights
    w_pairs /= sum_w
    w_bx /= sum_w
    w_by /= sum_w
    
    # Gradient of L w.r.t centers
    # dL/dc_k = sum_i w_i * (-s * d(d_i)/dc_k)
    # Note: obj = -1/s * L. So d(obj)/dc_k = -1/s * dL/dc_k = sum_i w_i * d(d_i)/dc_k
    
    # Wait, let's re-verify the sign.
    # soft_min = -1/s * L
    # We minimize -soft_min = 1/s * L.
    # So objective for minimize is 1/s * L.
    # Gradient of objective = 1/s * dL/dc_k.
    # dL/dc_k = sum_i w_i * (-s * d(d_i)/dc_k)
    # So Grad_obj = 1/s * sum_i w_i * (-s * d(d_i)/dc_k) = - sum_i w_i * d(d_i)/dc_k
    
    # Let's check:
    # If d_i decreases, L decreases (since exp(-s*d) increases? No.
    # d_i small -> -s*d_i large (less negative) -> exp large.
    # So L is large when d_i is small.
    # We want to INCREASE d_i.
    # So we want to DECREASE L.
    # Gradient of L points in direction of increasing L (decreasing d_i).
    # We want to go in direction of -Gradient(L).
    # Minimize 1/s * L. Gradient is 1/s * Grad_L.
    # So we should move in direction -Grad_L.
    
    # Let's compute d(d_i)/dc_k.
    # d_ij = (x_i - x_j)^2 + (y_i - y_j)^2
    # d(d_ij)/dx_i = 2(x_i - x_j)
    # d(d_ij)/dx_j = -2(x_i - x_j)
    
    # Accumulate gradients
    grad = np.zeros((N, 2))
    
    # Pairwise contributions
    # For each pair (i,j) with i<j, weight w_ij.
    # Contribution to grad[i] += w_ij * (2*(x_i-x_j), 2*(y_i-y_j))
    # Contribution to grad[j] += w_ij * (-2*(x_i-x_j), -2*(y_i-y_j))
    # Note: The sign in Grad_obj formula was - sum w_i d(d_i)/dc.
    # So we are summing - w_ij * d(d_ij)/dc.
    # d(d_ij)/dx_i = 2(x_i-x_j).
    # So grad_obj[x_i] += - w_ij * 2(x_i-x_j) = w_ij * 2(x_j-x_i).
    # This means if x_i < x_j, grad is positive, pushing x_i up (towards x_j)?
    # Wait. If x_i < x_j, they are close. We want to increase distance.
    # So x_i should move left (decrease) and x_j should move right (increase).
    # If x_i < x_j, x_j - x_i > 0.
    # grad_obj[x_i] += w * 2(x_j - x_i) > 0. This pushes x_i UP (towards x_j).
    # That's WRONG. We want to push them APART.
    
    # Let's re-evaluate.
    # We want to MAXIMIZE min dist.
    # Objective for minimize: -soft_min.
    # soft_min approximates min dist.
    # If we move points apart, min dist increases, soft_min increases, -soft_min decreases.
    # So gradient of -soft_min should point towards "together"?
    # No. Minimize f(x). If f is lower when x is "better", we move in direction -grad(f).
    # If "better" is larger dist, and f = -dist, then grad(f) = -1.
    # -grad(f) = 1. Move in + direction.
    
    # Let's stick to the formula:
    # f = 1/s * L
    # L = log(sum(exp(-s d_i)))
    # dL/dc_k = sum w_i (-s d(d_i)/dc_k)
    # df/dc_k = 1/s * dL/dc_k = - sum w_i d(d_i)/dc_k
    
    # For pair (i,j):
    # d(d_ij)/dx_i = 2(x_i - x_j)
    # Term in sum: - w_ij * 2(x_i - x_j) = 2 w_ij (x_j - x_i)
    
    # If x_i < x_j, x_j - x_i > 0. Grad is positive.
    # Minimize f. Move x_i in direction -grad.
    # -grad is negative. x_i decreases.
    # x_i moves left. x_j moves right.
    # Distance increases.
    # CORRECT.
    
    # So we accumulate:
    # grad[i] += w_ij * 2 * (x_j - x_i, y_j - y_i)  -- Wait, this is -d(d_i)/dc.
    # Let's just compute the vector V_ij = 2 * (x_j - x_i, y_j - y_i).
    # grad[i] += w_ij * V_ij
    # grad[j] += w_ij * (-V_ij)
    
    # Iterate over non-zero weights
    # For efficiency, we can do this with matrix operations or sparse.
    # N=200 is small enough for loops if optimized, but matrix is better.
    # However, w_pairs is dense (20k non-zeros).
    
    # Let's use a loop over the flat indices of the upper triangle.
    # 20k iterations in Python is ~10ms. Acceptable for 150s budget if we do ~1000 iters?
    # 1000 * 10ms = 10s. Yes.
    
    # Get indices
    i_idx, j_idx = np.where(mask)
    
    w_flat = w_pairs[i_idx, j_idx]
    
    # Vectors
    # dx_ij = x_j - x_i
    dx_ij = x[j_idx] - x[i_idx]
    dy_ij = y[j_idx] - y[i_idx]
    
    # Weighted vectors
    wx = w_flat * 2.0 * dx_ij
    wy = w_flat * 2.0 * dy_ij
    
    # Accumulate to grad[i] += wx
    # grad[j] -= wx
    # Use np.add.at for unbuffered accumulation
    
    np.add.at(grad, (i_idx, 0), wx)
    np.add.at(grad, (j_idx, 0), -wx)
    np.add.at(grad, (i_idx, 1), wy)
    np.add.at(grad, (j_idx, 1), -wy)
    
    # Boundary contributions
    # d_bx_i = (0.5 - |x_i|)^2
    # d(d_bx_i)/dx_i = 2(0.5 - |x_i|) * d(0.5 - |x_i|)/dx_i
    # d(0.5 - |x_i|)/dx_i = -sign(x_i)
    # So d(d_bx_i)/dx_i = -2(0.5 - |x_i|) sign(x_i)
    
    # Term in grad_obj: - w_bx_i * d(d_bx_i)/dx_i = 2 w_bx_i (0.5 - |x_i|) sign(x_i)
    
    # If x_i > 0, sign=1. Term = 2 w (0.5 - x_i).
    # If x_i < 0, sign=-1. Term = -2 w (0.5 + x_i) = 2 w (x_i - 0.5) = -2 w (0.5 - |x_i|) * (-1)?
    # Let's check: x_i = -0.4. |x_i|=0.4. 0.5-|x_i|=0.1.
    # d/dx (0.5 - |-0.4|)^2.
    # Let u = 0.5 - |x|. u' = -sign(x).
    # d/dx u^2 = 2u u' = 2(0.5-|x|)(-sign(x)).
    # If x<0, sign(x)=-1. u' = 1.
    # d/dx = 2(0.5-|x|). Positive.
    # Grad_obj term = - w * d/dx.
    # So grad_obj[x_i] -= w * 2(0.5-|x_i|).
    # If x_i = -0.4, grad is negative. Move x_i left (more negative).
    # Distance to wall 0.5 - |-0.4| = 0.1.
    # Moving to -0.41, dist = 0.09.
    # We want to INCREASE dist.
    # So we should move AWAY from wall.
    # Wall is at -0.5. x_i is -0.4. Away is towards 0.
    # So x_i should increase.
    # Grad should be negative?
    # Minimize f. Move -grad.
    # If grad is negative, -grad is positive. x_i increases.
    # Correct.
    
    # Formula:
    # grad_x_bx = - w_bx * 2 * (0.5 - |x|) * (-sign(x)) = 2 w_bx (0.5 - |x|) sign(x)
    # Wait.
    # d(d_bx)/dx = 2(0.5-|x|)(-sign(x))
    # Grad_obj = - sum w d(d)/dx = - w * 2(0.5-|x|)(-sign(x)) = 2 w (0.5-|x|) sign(x)
    
    # If x > 0: sign=1. Grad = 2w(0.5-x). Positive.
    # Move -grad (negative). x decreases. Away from right wall (0.5). Correct.
    # If x < 0: sign=-1. Grad = -2w(0.5+|x|) = 2w(x-0.5). Negative.
    # Move -grad (positive). x increases. Away from left wall (-0.5). Correct.
    
    sign_x = np.sign(x)
    # Handle x=0. sign(0)=0. Gradient 0. Fine.
    grad_bx = 2.0 * w_bx * (0.5 - np.abs(x)) * sign_x
    grad_by = 2.0 * w_by * (0.5 - np.abs(y)) * sign_x # Wait, sign_y
    
    sign_y = np.sign(y)
    grad_by = 2.0 * w_by * (0.5 - np.abs(y)) * sign_y
    
    grad[:, 0] += grad_bx
    grad[:, 1] += grad_by
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # Initial guess: Hexagonal grid
    def hex_init(n):
        s = np.sqrt(2.0 / (n * np.sqrt(3.0)))
        max_rows = int(1.0 / (s * np.sqrt(3.0)/2.0)) + 1
        max_cols = int(1.0 / s) + 1
        
        points = []
        for r in range(max_rows):
            y = (r - max_rows/2.0 + 0.5) * s * np.sqrt(3.0)/2.0
            offset = 0.0 if r % 2 == 0 else s/2.0
            for c in range(max_cols):
                x = (c - max_cols/2.0 + 0.5) * s + offset
                points.append([x, y])
                
        points = np.array(points)
        mask = (points[:, 0] > -0.5) & (points[:, 0] < 0.5) & (points[:, 1] > -0.5) & (points[:, 1] < 0.5)
        points = points[mask]
        
        if len(points) >= n:
            indices = np.random.choice(len(points), n, replace=False)
            points = points[indices]
        else:
            extra = np.random.uniform(-0.4, 0.4, (n - len(points), 2))
            points = np.vstack([points, extra])
            
        return points

    centers = hex_init(N)
    centers = np.clip(centers, -0.49, 0.49) # Keep strictly inside
    
    best_score = 0.0
    best_centers = centers.copy()
    
    def compute_real_score(c):
        # Fast real score for tracking best
        # Use cKDTree for speed
        tree = cKDTree(c)
        # Query pairs with r = current_best * 2 + epsilon
        # But we don't know current best yet.
        # For N=200, just compute full min dist.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        min_b = np.min(np.minimum(dist_x, dist_y))
        
        return min(min_d / 2.0, min_b)
    
    current_score = compute_real_score(centers)
    best_score = current_score
    best_centers = centers.copy()
    
    # Optimization Loop
    # We will run several rounds of:
    # 1. Gradient Descent (L-BFGS-B)
    # 2. Kick
    
    sharpness = 50.0
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Main Loop
    while time.time() < deadline:
        # Run L-BFGS-B
        # Use a limited number of iterations to keep it fast and responsive to kicks
        try:
            res = minimize(
                _objective_and_grad, 
                centers.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-5}
            )
            centers = res.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
        except Exception as e:
            # If optimization fails, break and try kick
            pass
            
        # Evaluate
        current_score = compute_real_score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Kick: Perturb the points involved in the closest pairs
        # Find closest pairs
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Get indices of closest pairs
        # Flatten and find smallest
        dists_flat = dists.flatten()
        # We want the smallest few
        # Argsort is O(N^2 log N). 40k log 40k ~ 600k ops. Fast.
        idx_flat = np.argpartition(dists_flat, 10)[:10]
        
        # Map back to i, j
        i_idx = idx_flat // N
        j_idx = idx_flat % N
        
        # Get unique points involved
        points_to_kick = np.unique(np.concatenate([i_idx, j_idx]))
        
        # Perturb them
        # Random small displacement
        noise = np.random.normal(0, 0.01, (len(points_to_kick), 2))
        centers[points_to_kick] += noise
        centers = np.clip(centers, -0.5, 0.5)
        
        # Increase sharpness slightly to focus on tighter constraints?
        # Or keep it constant.
        # Let's keep it constant for stability.
        
    return best_centers
# EVOLVE-BLOCK-END
