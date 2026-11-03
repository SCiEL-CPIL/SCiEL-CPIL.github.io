"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the score (minimum radius) for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = wall_dists.min()
    if min_wall <= 0:
        return -np.inf
    # Pairwise distances
    # Use a chunked approach to avoid memory issues for large N, but N=100 is fine
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_wall, min_pair)

def _objective_and_grad(x, active_pairs, wall_active):
    """
    Objective: minimize -r (maximize r)
    Variables: x (N*3), r (1)
    Constraints handled by SLSQP via bounds and equality/inequality?
    Actually, SLSQP handles bounds easily.
    We want to maximize r subject to:
    ||c_i - c_j|| >= 2r for i<j
    0.5 - |c_ik| >= r
    
    This is not a standard convex form for SLSQP directly because of the absolute values and quadratic constraints.
    However, we can reformulate using auxiliary variables or just use gradient-based methods on the "soft" min or use the active set.
    
    Let's stick to a simpler heuristic: 
    1. Start with a good grid.
    2. Identify "tight" constraints (pairs with dist < 2r + eps, walls with 0.5-|c| < r + eps).
    3. Solve a local optimization problem maximizing r subject to these active constraints being equalities (or close).
    
    Alternatively, use a potential energy method which is robust.
    Energy E(r) = sum_{i<j} (1/||c_i-c_j||^2) + sum_i (1/(0.5-|c_ik|)^2)
    If we minimize energy, points move apart.
    But we want to maximize the *minimum* distance.
    
    Let's try a direct approach: 
    Start with a grid.
    Iteratively:
    1. Compute current r.
    2. Find the most violated constraint (smallest distance to wall or between points).
    3. Apply a gradient step to increase that specific distance, while trying to maintain others.
    
    A better method: "Push and Shove" or simple gradient ascent on the soft-min.
    Soft-min of distances: L = -log(sum(exp(-d_i / tau)))
    Gradient of this w.r.t positions can be computed.
    
    Let's implement a gradient ascent on the soft-min of the minimum radius.
    """
    pass

def _soft_min_radius(centers, tau):
    """
    Compute soft-min of (min wall dist, min pair dist/2) and its gradient.
    Returns: value, grad (N, 3)
    """
    centers = np.asarray(centers, dtype=np.float64)
    N_pts = len(centers)
    
    # Wall distances: d_w_i = 0.5 - |c_ik|
    # We want to maximize min(d_w).
    # Soft max of d_w: log(sum(exp(d_w / tau))) * tau
    # Let's compute soft-max of the "clearances".
    
    # Wall clearances
    # c is (N, 3). 
    # dist to left wall: c[:, 0] + 0.5
    # dist to right wall: 0.5 - c[:, 0]
    # etc.
    # There are 6 walls.
    # d_w shape (N, 6)
    d_w = np.zeros((N_pts, 6))
    d_w[:, 0] = centers[:, 0] + 0.5  # x > -0.5
    d_w[:, 1] = 0.5 - centers[:, 0]  # x < 0.5
    d_w[:, 2] = centers[:, 1] + 0.5  # y > -0.5
    d_w[:, 3] = 0.5 - centers[:, 1]  # y < 0.5
    d_w[:, 4] = centers[:, 2] + 0.5  # z > -0.5
    d_w[:, 5] = 0.5 - centers[:, 2]  # z < 0.5
    
    # Pair distances / 2
    # d_p shape (N, N)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_p = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(d_p, np.inf)
    
    # Soft max of all these values?
    # We want to maximize min(all d_w, all d_p).
    # Let S be the set of all clearances.
    # Soft max: tau * log(sum(exp(s_i / tau)))
    
    # To save memory and time, we can combine.
    # But N=100, so 6*100 + 100*99/2 = 600 + 4950 = 5550 values.
    # This is small.
    
    all_clearances = np.concatenate([d_w.flatten(), d_p[d_p < np.inf].flatten()])
    
    # Numerical stability for log-sum-exp
    max_c = np.max(all_clearances)
    exp_c = np.exp((all_clearances - max_c) / tau)
    sum_exp = np.sum(exp_c)
    soft_max_val = max_c + tau * np.log(sum_exp)
    
    # Gradient
    # d(soft_max)/d(s_i) = exp(s_i / tau) / sum(exp(s_j / tau)) = exp(s_i - max_c) / sum_exp
    # Let weights w_i = exp(s_i - max_c) / sum_exp
    
    weights = np.zeros_like(all_clearances)
    weights[:] = exp_c / sum_exp
    
    # Now, map weights back to centers.
    # grad_c = sum_i w_i * grad(s_i) / c_i
    
    grad = np.zeros((N_pts, 3))
    
    # Wall contributions
    w_w = weights[:N_pts * 6].reshape(N_pts, 6)
    # s_0 = c_x + 0.5 -> grad_x = 1
    # s_1 = 0.5 - c_x -> grad_x = -1
    # s_2 = c_y + 0.5 -> grad_y = 1
    # s_3 = 0.5 - c_y -> grad_y = -1
    # s_4 = c_z + 0.5 -> grad_z = 1
    # s_5 = 0.5 - c_z -> grad_z = -1
    
    grad[:, 0] += w_w[:, 0] - w_w[:, 1]
    grad[:, 1] += w_w[:, 2] - w_w[:, 3]
    grad[:, 2] += w_w[:, 4] - w_w[:, 5]
    
    # Pair contributions
    # d_p[i,j] = ||c_i - c_j|| / 2
    # grad(d_p[i,j]) w.r.t c_i = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (4 * d_p[i,j])
    # Note: d_p is symmetric.
    
    w_p = weights[N_pts * 6:].reshape(N_pts, N_pts)
    # The flattened d_p had inf on diagonal, so those weights are 0 (exp(-inf)=0).
    # But we need to be careful with the shape.
    # d_p was (N, N). We flattened it excluding diagonal?
    # In `all_clearances`, we did `d_p[d_p < np.inf].flatten()`.
    # This order is row-major, skipping diagonal.
    # It is easier to recompute the gradient using the full matrix logic if we can map weights back.
    # Let's create a full weight matrix for pairs.
    w_p_full = np.zeros((N_pts, N_pts))
    # We need to map the flattened vector back to the matrix.
    # This is tricky. Let's just recompute the soft max on the full matrix including diagonal (with 0 weight) or handle it carefully.
    
    # Alternative: Compute soft max on full matrices and sum.
    # Soft max is not linear, so we can't just sum soft maxes.
    # We must do it on the concatenated vector.
    
    # Let's reconstruct the pair weights matrix.
    # The order in `d_p[d_p < np.inf]` is row by row, skipping i=j.
    idx = 0
    for i in range(N_pts):
        for j in range(N_pts):
            if i != j:
                w_p_full[i, j] = weights[N_pts * 6 + idx]
                idx += 1
    
    # Gradient from pairs:
    # For each pair (i, j), contribution to grad[i] is w_p_full[i, j] * (c_i - c_j) / (4 * d_p[i, j])
    # And to grad[j] is w_p_full[j, i] * (c_j - c_i) / (4 * d_p[j, i])
    
    # Vectorized:
    # diff = c_i - c_j
    # dist = ||diff||
    # term = (c_i - c_j) / (2 * dist)  <-- wait, d_p = dist/2. grad(d_p) = (c_i-c_j)/(2*dist).
    # So term = (c_i - c_j) / (2 * dist).
    
    # We have w_p_full[i, j].
    # grad[i] += sum_j w_p_full[i, j] * (c_i - c_j) / (2 * dist_ij)
    
    # Let's compute this efficiently.
    # dist_matrix = 2 * d_p (with inf on diag)
    dist_mat = 2.0 * d_p
    # Avoid division by zero on diagonal (weights are 0 there anyway, but inf*0 is nan)
    # Set diagonal to 1 to avoid nan, since weight is 0.
    np.fill_diagonal(dist_mat, 1.0)
    
    # factor[i, j] = w_p_full[i, j] / (2 * dist_mat[i, j])
    # Note: d_p = dist/2, so dist = 2*d_p.
    # grad(d_p) = (c_i - c_j) / (2 * dist) = (c_i - c_j) / (4 * d_p).
    # My formula above: (c_i - c_j) / (2 * dist). Correct.
    
    # We want grad[i] = sum_j w_p_full[i, j] * (c_i - c_j) * (1 / (2 * dist_ij))
    
    # Let A[i, j] = w_p_full[i, j] / (2 * dist_ij)
    A = w_p_full / (2.0 * dist_mat)
    np.fill_diagonal(A, 0.0)
    
    # grad[i] = sum_j A[i, j] * (c_i - c_j)
    #          = c_i * sum_j A[i, j] - sum_j A[i, j] * c_j
    
    sum_A = A.sum(axis=1, keepdims=True) # (N, 1)
    # term1 = c_i * sum_A[i]
    term1 = centers * sum_A
    
    # term2[i] = sum_j A[i, j] * c_j
    term2 = A @ centers # (N, 3)
    
    grad_pair = term1 - term2
    grad += grad_pair
    
    return soft_max_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Parameters
    tau = 0.01  # Temperature for soft-min
    lr = 0.01   # Learning rate
    
    # Optimization loop
    for step in range(10000):
        if time.time() > deadline:
            break
            
        # Compute soft-min radius and gradient
        val, grad = _soft_min_radius(centers, tau)
        
        # Gradient ascent
        centers += lr * grad
        
        # Project back to cube? 
        # The soft-min penalizes going outside, but it's not a hard constraint.
        # We should clip to [-0.5 + eps, 0.5 - eps] to ensure validity if we use this directly.
        # However, clipping can mess up the gradient.
        # Better: Add a hard constraint penalty or just clip and recompute.
        # Since we want to maximize min distance, and wall distance is part of the objective,
        # the gradient will push away from walls.
        # But to be safe, let's clip.
        centers = np.clip(centers, -0.499, 0.499)
        
        # Evaluate true radius
        current_radius = _compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
        # Anneal tau?
        # If tau is too small, it's like hard min, which is non-differentiable.
        # If tau is large, it's an average.
        # Start with moderate tau, decrease it.
        tau *= 0.999
        
        # Adaptive learning rate?
        # If no improvement for a while, reduce lr.
        # (Simple implementation)
        
    # Final polish: Local search on the best center found
    # We can do a few more steps with smaller tau and lr
    tau = 0.001
    lr = 0.001
    centers = best_centers.copy()
    
    for step in range(1000):
        if time.time() > deadline:
            break
        val, grad = _soft_min_radius(centers, tau)
        centers += lr * grad
        centers = np.clip(centers, -0.499, 0.499)
        
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
