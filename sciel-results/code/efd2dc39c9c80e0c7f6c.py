"""
Smooth soft-min optimization for packing 100 equal spheres in a unit cube.

Uses a log-sum-exp approximation of the minimum distance (soft-min) to create
a differentiable objective function. This allows the use of L-BFGS-B to
maximize the minimal distance (radius). An annealing schedule on the
sharpness parameter beta is used to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE_MARGIN = 5.0  # Safety margin before the 150s kill


def _exact_radius(pts):
    """Compute the exact maximum common radius for a set of centers."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    
    # Distance to walls
    # pts are in [-0.5, 0.5]
    wall_dists = HALF - np.max(np.abs(pts), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the minimum of (pair dist, 2 * wall dist)
    # Wait, the constraint is |c_i - c_j| >= 2r and |c_ik| <= 0.5 - r.
    # So 2r <= |c_i - c_j| and 2r <= 1 - 2|c_ik| ? No.
    # Center is at c. Sphere radius r.
    # Distance to wall is 0.5 - |c| (if c>0) or 0.5 + c (if c<0). i.e., 0.5 - |c|.
    # Constraint: r <= 0.5 - |c_ik|.
    # So 2r <= 1 - 2|c_ik| is incorrect.
    # The radius r must satisfy r <= 0.5 - max_k |c_ik|.
    # And r <= 0.5 * min_{i<j} |c_i - c_j|.
    
    r_pair = 0.5 * min_pair_dist
    r_wall = min_wall_dist
    
    r = min(r_pair, r_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


def _soft_min_objective(x, beta):
    """
    Objective function to minimize: -soft_min(distances).
    x: flattened array of shape (N*3,)
    beta: sharpness parameter for the soft-min.
    Returns (value, gradient).
    """
    pts = x.reshape(N, 3)
    
    # 1. Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1e10) # Avoid sqrt(0) issues, though we handle indices
    dists = np.sqrt(d2)
    
    # Get upper triangle indices to avoid double counting and self
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Wall distances
    # Distance from center to nearest wall
    # For a point c, dist to wall is 0.5 - |c|
    wall_dists = HALF - np.max(np.abs(pts), axis=1)
    
    # Combine all distances into one array
    # Note: Pair distances contribute to radius as d/2.
    # Wall distances contribute to radius as d.
    # To unify, we can work with "radius-equivalent" distances:
    # r_pair_eq = pair_dists / 2
    # r_wall_eq = wall_dists
    # We want to maximize min(r_pair_eq, r_wall_eq).
    # So we minimize -soft_min(r_pair_eq, r_wall_eq).
    
    r_vals = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min calculation
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, use log-sum-exp trick:
    # m = min(x)
    # soft_min = m - 1/beta * log(sum(exp(-beta * (x - m))))
    
    m = np.min(r_vals)
    # If m is very small or negative, we might have issues, but r_vals should be positive
    # Clamp r_vals to a small positive number to avoid log(0) or negative distances due to numerical error
    r_vals_safe = np.maximum(r_vals, 1e-10)
    m = np.min(r_vals_safe)
    
    exp_terms = np.exp(-beta * (r_vals_safe - m))
    sum_exp = np.sum(exp_terms)
    log_sum_exp = np.log(sum_exp)
    
    soft_min_val = m - (1.0 / beta) * log_sum_exp
    
    # We want to maximize soft_min_val, so we minimize -soft_min_val
    obj_val = -soft_min_val
    
    # Gradient
    # d/dx soft_min(x) = -1/beta * d/dx log(sum(exp(-beta * x)))
    # = -1/beta * (1/sum(exp)) * sum(-beta * exp * dx)
    # = sum(exp) / sum(exp) * dx  (Wait, let's derive carefully)
    
    # Let S = sum(exp(-beta * (r_k - m)))
    # soft_min = m - 1/beta * log(S)
    # Note m depends on x too, but for soft-min approximation, the gradient is often
    # approximated by the weighted average of the gradients of the individual terms,
    # where weights are p_k = exp(-beta(r_k - m)) / S.
    # Actually, the exact gradient of the log-sum-exp soft-min is:
    # grad soft_min = sum_k p_k * grad r_k
    # where p_k = exp(-beta(r_k - m)) / S.
    
    # Let's verify:
    # f(x) = -1/beta log(sum exp(-beta r_k(x)))  (ignoring m for a moment, or assuming m is constant for gradient direction)
    # Actually, the standard soft-min gradient is indeed the softmax-weighted average of the gradients of the arguments.
    
    weights = exp_terms / sum_exp
    
    # Gradient of r_vals w.r.t x
    # r_vals consists of pair_dists/2 and wall_dists.
    
    # 1. Pairwise part
    # r_pair = 0.5 * |c_i - c_j|
    # d r_pair / d c_i = 0.5 * (c_i - c_j) / |c_i - c_j|
    # d r_pair / d c_j = 0.5 * (c_j - c_i) / |c_i - c_j|
    
    grad_x = np.zeros_like(pts)
    
    # We have weights for pairs (len i_idx) and walls (N)
    w_pairs = weights[:len(i_idx)]
    w_walls = weights[len(i_idx):]
    
    # Pair gradients
    # diff_ij = c_i - c_j
    # We need to sum w_k * grad r_k
    # For each pair (i, j), contribution to c_i is w_k * 0.5 * (c_i - c_j) / d_ij
    # Contribution to c_j is w_k * 0.5 * (c_j - c_i) / d_ij
    
    # Vectorized approach:
    # Create a matrix of weights W (N, N) with 0 on diagonal
    # W[i, j] = w_pair for pair (i, j)
    # Then grad_pair_i = 0.5 * sum_j W[i, j] * (c_i - c_j) / d_ij
    
    # This is O(N^2) which is fine for N=100.
    
    # Reconstruct full distance matrix for gradient calculation
    # We already have dists (N, N)
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    # Unit vectors
    # u_ij = (c_i - c_j) / d_ij
    u = diff / dists_safe[:, :, None]
    
    # Weight matrix
    W_mat = np.zeros((N, N))
    W_mat[i_idx, j_idx] = w_pairs
    W_mat[j_idx, i_idx] = w_pairs # Symmetric
    
    # grad_i = 0.5 * sum_j W_mat[i, j] * u_ij
    # Note: u_ij = -u_ji
    # grad_pair = 0.5 * (W_mat * u).sum(axis=1)
    # Wait, the term is (c_i - c_j)/d_ij which is u_ij.
    # So grad_i = 0.5 * sum_j W_mat[i, j] * u_ij
    
    grad_pair_part = 0.5 * np.einsum('ij,ijk->ik', W_mat, u)
    
    # 2. Wall gradients
    # r_wall_i = 0.5 - |c_ik|
    # d r_wall_i / d c_ik = -sign(c_ik)
    # So grad_wall_i = -sign(c_i)
    
    # We need to sum w_wall_i * grad_wall_i for each coordinate?
    # No, r_wall is a scalar per point i.
    # The gradient of the sum of soft-min terms is sum_k w_k grad r_k.
    # For wall term k (corresponding to point i, coord k is not separate, it's max coord),
    # wait, r_wall_i = min_k (0.5 - |c_ik|) ?
    # No, the constraint is r <= 0.5 - |c_ik| for ALL k.
    # So the limiting wall distance is min_k (0.5 - |c_ik|).
    # In my `r_vals` construction, I used `wall_dists = HALF - np.max(np.abs(pts), axis=1)`.
    # This is correct: the distance to the nearest wall for point i is 0.5 - max_k |c_ik|.
    # The gradient of `max_k |c_ik|` is `sign(c_ik) * e_k` where k is the argmax.
    # So gradient of `r_wall_i` is `-sign(c_ik) * e_k`.
    
    # We have w_walls[i] as the weight for point i's wall constraint.
    # grad_wall_part[i] = w_walls[i] * (-sign(c_i[argmax_k |c_ik|]) * e_k)
    
    abs_pts = np.abs(pts)
    max_coord_idx = np.argmax(abs_pts, axis=1)
    
    # Create a one-hot vector for the max coordinate
    # shape (N, 3)
    one_hot = np.zeros_like(pts)
    one_hot[np.arange(N), max_coord_idx] = 1.0
    
    # Gradient of r_wall_i w.r.t c_i is -sign(c_i[max_coord]) * one_hot
    # sign(c) is 0 if c=0, but we can ignore that measure-zero case.
    signs = np.sign(pts)
    # We only want the sign of the component that is max
    # signs_max = signs * one_hot (elementwise)
    grad_wall_part = -w_walls[:, None] * signs * one_hot
    
    # Total gradient of soft_min
    grad_soft_min = grad_pair_part + grad_wall_part
    
    # We are minimizing -soft_min, so gradient is -grad_soft_min
    grad_obj = -grad_soft_min
    
    return obj_val, grad_obj.ravel()


def _optimize_beta_schedule(init_pts, deadline, betas):
    """
    Run L-BFGS-B with a schedule of beta values.
    Returns best points found.
    """
    current_pts = init_pts.copy()
    best_pts = current_pts.copy()
    best_r = _exact_radius(best_pts)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Define objective and gradient for this beta
        def fun(x):
            val, grad = _soft_min_objective(x, beta)
            return val
        
        def jac(x):
            val, grad = _soft_min_objective(x, beta)
            return grad
        
        # L-BFGS-B with bounds
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-HALF, HALF)] * (N * 3)
        
        try:
            res = minimize(
                fun, 
                current_pts.ravel(), 
                jac=jac, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            
            cand_pts = res.x.reshape(N, 3)
            # Ensure points are within bounds (L-BFGS-B should handle this, but just in case)
            cand_pts = np.clip(cand_pts, -HALF, HALF)
            
            r = _exact_radius(cand_pts)
            if r > best_r:
                best_r = r
                best_pts = cand_pts.copy()
            
            # Use the result as the start for the next beta
            current_pts = cand_pts
            
        except Exception as e:
            # If optimization fails, keep current points
            pass
            
    return best_pts, best_r


def _init_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_random(rng):
    """Initialize with random points."""
    return (rng.random((N, 3)) - 0.5) * 0.9


def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Global best
    best_pts = _init_grid()
    best_r = _exact_radius(best_pts)
    
    # Beta schedule: Start smooth, end sharp
    # Small beta: broad minima, easier to escape. Large beta: sharp, converges to local min of min-dist.
    betas = [5, 10, 20, 40, 80, 150]
    
    # Initializations: Grid + Random
    inits = [best_pts]
    for _ in range(5):
        inits.append(_init_random(rng))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        p, r = _optimize_beta_schedule(init, deadline, betas)
        if r > best_r:
            best_r = r
            best_pts = p
    
    # Use spare time for more random restarts with higher beta polish
    guard = 0
    while time.time() < deadline - 2.0 and guard < 10:
        guard += 1
        start = _init_random(rng)
        # Use a shorter, sharper beta schedule for quick polishing
        betas_short = [20, 50, 100]
        p, r = _optimize_beta_schedule(start, deadline, betas_short)
        if r > best_r:
            best_r = r
            best_pts = p
    
    # Final safety check
    out = np.clip(np.asarray(best_pts, dtype=np.float64), -HALF, HALF)
    if not np.all(np.isfinite(out)):
        out = _init_grid()
        
    return out
