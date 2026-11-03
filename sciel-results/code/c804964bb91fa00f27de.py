"""Structural change: Gradient-based optimization with soft-min objective and inflation phase."""
N = 200
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 115.0  # Safety margin below 120s kill
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _DEADLINE

def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)

def _soft_min_neg(x, beta):
    """
    Compute -soft_min(x, beta) and its gradient.
    soft_min(x, beta) = -1/beta * log(sum(exp(-beta * x_i)))
    We want to maximize min(x), so we minimize -min(x).
    f(x) = -soft_min(x, beta) = 1/beta * log(sum(exp(-beta * x_i)))
    grad_f = -exp(-beta * x_i) / sum(exp(-beta * x_j))
    """
    # Numerical stability: subtract max
    x = np.asarray(x)
    if x.size == 0:
        return 0.0, np.array([])
    
    # To avoid overflow in exp(-beta * x), we can use the log-sum-exp trick.
    # log(sum(exp(-beta * x))) = -beta * max(x) + log(sum(exp(-beta * (x - max(x)))))
    # Wait, standard log-sum-exp is for sum(exp(x)).
    # Let y = -beta * x. We want log(sum(exp(y))).
    # max_y = max(y) = -beta * min(x).
    # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
    
    y = -beta * x
    max_y = np.max(y)
    # If max_y is very large positive, exp(y-max_y) is safe (<= 1).
    # If max_y is very large negative, it's also safe relative to max.
    
    # However, if beta is large and x has large negative values, y is large positive.
    # If x has large positive values, y is large negative.
    
    # Let's just use the stable form.
    exp_y = np.exp(y - max_y)
    sum_exp_y = np.sum(exp_y)
    
    val = (max_y + np.log(sum_exp_y)) / beta  # This is soft_min? No.
    # f(x) = 1/beta * log(sum(exp(-beta * x)))
    # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
    # So f(x) = (max_y + log(sum_exp_y)) / beta
    
    # Gradient of f wrt x_i:
    # df/dx_i = 1/beta * (1/sum_exp_y) * d/dx_i (sum(exp(y)))
    # = 1/beta * (1/sum_exp_y) * sum_j [exp(y_j) * (-beta) * delta_ij]
    # = 1/beta * (1/sum_exp_y) * exp(y_i) * (-beta)
    # = -exp(y_i) / sum_exp_y
    
    # But we computed exp(y - max_y).
    # exp(y_i) = exp(y_i - max_y) * exp(max_y)
    # sum_exp_y = sum(exp(y_j - max_y)) * exp(max_y)
    # So ratio is exp(y_i - max_y) / sum(exp(y_j - max_y))
    
    weights = exp_y / sum_exp_y
    grad = -weights
    
    return val, grad

def _objective_and_grad(centers_flat, beta):
    """
    Objective: Minimize -min(distances/2, boundaries)
    Using soft-min approximation.
    """
    centers = centers_flat.reshape(-1, 2)
    
    # 1. Pairwise distances
    # diff[i,j] = c_i - c_j
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid sqrt for gradient? No, we need dists.
    # dists = sqrt(dists_sq)
    # Gradient of dist wrt c_i is (c_i - c_j) / dist
    
    # We only care about the minimum, so we can use a soft min over all pairs.
    # But N^2 is 40k. That's manageable.
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_sq_masked = np.where(mask, dists_sq, np.inf)
    
    # We want to minimize -min(dists/2).
    # Let x = dists/2. We want -soft_min(x, beta).
    # x = 0.5 * sqrt(dists_sq)
    
    # To save time, we can compute soft min on dists_sq?
    # min(sqrt(d)) is monotonic with min(d).
    # But the gradient changes.
    # Let's stick to dists.
    
    # Efficiently compute dists
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Flatten dists for soft min
    dists_flat = dists[mask]
    
    # Soft min of dists/2
    # val, grad_dists = _soft_min_neg(dists_flat / 2.0, beta)
    
    # 2. Boundaries
    # b_i = min(0.5 - |c_ik|)
    # We need soft min of all boundary distances.
    
    # Let's construct the full vector of constraints:
    # C = [dists_flat/2, boundary_x, boundary_y]
    
    # Boundary x: 0.5 - |x_i|
    # Boundary y: 0.5 - |y_i|
    
    # This is getting complex to vectorize the gradient for the combined soft min.
    # Alternative: Separate soft mins? No, min(A, B) is not min(A)+min(B).
    
    # Let's use a simpler approach:
    # Objective = max(0, max(0.5 - dist/2, 0.5 - boundary))
    # This is non-smooth.
    
    # Let's use the soft-min on the "violations" or "margins".
    # We want to maximize r.
    # r <= dist_ij / 2
    # r <= 0.5 - |c_ik|
    
    # Let's define the objective as:
    # f(r, c) = sum (max(0, r - dist/2)^2 + max(0, r - boundary)^2)
    # This is smooth for r > 0.
    # But we want to optimize c for a fixed r? Or optimize both?
    # If we optimize both, we can start with small r and increase.
    
    # Better strategy:
    # 1. Fix r = 0.
    # 2. Minimize "energy" = sum (1/d^2) for pairs + sum (1/boundary^2) for walls.
    # This is the force-directed approach but with a smoother potential.
    
    # Let's try the "Inflation" strategy directly on the current grid.
    # Start with a valid packing (grid).
    # Compute current r.
    # Try to increase r slightly, and adjust centers to satisfy constraints.
    # This is a feasibility problem.
    
    # Let's implement a simpler gradient descent on the "min distance" metric.
    # Use a soft-min over the *tightest* constraints.
    
    # To keep it simple and robust:
    # Use scipy.optimize.minimize on the centers, with objective = -soft_min(distances/2, boundaries, beta).
    
    # We need to compute the gradient of the combined soft min.
    
    # Let's construct the array of all "margins":
    # margins = [dists_flat/2, 0.5 - abs(x), 0.5 - abs(y)]
    
    # N=200. Pairs ~ 20k. Boundaries ~ 400. Total ~ 20k.
    # This is small enough for vectorization.
    
    # Compute margins
    margins = np.concatenate([
        dists_flat / 2.0,
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ])
    
    # Soft min
    val, grad_margins = _soft_min_neg(margins, beta)
    
    # Gradient wrt centers
    grad_centers = np.zeros_like(centers)
    
    # 1. Pairwise distances
    # grad_margins[0:K] corresponds to dists_flat/2
    # d(dist/2)/dc_i = (c_i - c_j) / (2 * dist)
    # Note: dists_flat is upper triangle? No, my mask is full.
    # dists is symmetric.
    # Let's be careful with indices.
    
    # It's easier to compute the gradient of the soft min of the *squared* distances?
    # No, let's stick to the direct gradient.
    
    # Re-compute dists and diffs to map gradients back.
    # diff[i,j] = c_i - c_j
    # dist_ij = |diff|
    # margin_ij = dist_ij / 2
    # d(margin_ij)/dc_i = diff / (2 * dist)
    # d(margin_ij)/dc_j = -diff / (2 * dist)
    
    # We have grad_margins for the flattened margins.
    # The first K elements are pairwise.
    K_pairs = dists_flat.size
    grad_pairs = grad_margins[:K_pairs]
    
    # We need to map grad_pairs back to the (N, N) matrix.
    # dists_flat was created by dists[mask].
    # So we can create a (N, N) matrix G_pairs and fill it.
    G_pairs = np.zeros((N, N))
    G_pairs[mask] = grad_pairs
    
    # Now, grad_centers[i] = sum_j G_pairs[i,j] * d(margin_ij)/dc_i
    # d(margin_ij)/dc_i = diff[i,j] / (2 * dists[i,j])
    
    # Let's compute this.
    # diff is (N, N, 2)
    # dists is (N, N)
    
    # Avoid division by zero on diagonal (masked out anyway, but dists diag is inf)
    inv_dists = np.zeros_like(dists)
    mask_inv = mask & (dists > 1e-12)
    inv_dists[mask_inv] = 1.0 / (2.0 * dists[mask_inv])
    
    # grad_contrib[i] = sum_j G_pairs[i,j] * diff[i,j] * inv_dists[i,j]
    # G_pairs[i,j] is a scalar. diff[i,j] is a vector.
    
    # We can do this with einsum or broadcasting.
    # (N, N) * (N, N, 2) -> sum over j -> (N, 2)
    
    # Note: G_pairs is symmetric?
    # soft_min is symmetric. grad_margins is symmetric?
    # soft_min(x) = -1/beta log(sum exp(-beta x)).
    # grad_i = -exp(-beta x_i) / sum exp(-beta x_j).
    # Since x_ij = x_ji, grad_i = grad_j.
    # So G_pairs is symmetric.
    
    # grad_centers[i] = sum_j G_pairs[i,j] * (c_i - c_j) / (2 * dist_ij)
    
    # Let's compute this efficiently.
    # term = G_pairs[:, :, np.newaxis] * diff * inv_dists[:, :, np.newaxis]
    # grad_centers = np.sum(term, axis=1)
    
    # This creates an (N, N, 2) array, which is 200*200*2 = 80k floats. Fine.
    
    term = G_pairs[:, :, np.newaxis] * diff * inv_dists[:, :, np.newaxis]
    grad_centers += np.sum(term, axis=1)
    
    # 2. Boundaries
    # grad_margins[K_pairs:K_pairs+N] for x
    # grad_margins[K_pairs+N:K_pairs+2N] for y
    
    grad_x = grad_margins[K_pairs:K_pairs+N]
    grad_y = grad_margins[K_pairs+N:K_pairs+2*N]
    
    # d(0.5 - |x|)/dx = -sign(x)
    # d(0.5 - |y|)/dy = -sign(y)
    
    # If x == 0, gradient is 0 (subgradient).
    signs_x = np.sign(centers[:, 0])
    signs_y = np.sign(centers[:, 1])
    
    grad_centers[:, 0] += grad_x * (-signs_x)
    grad_centers[:, 1] += grad_y * (-signs_y)
    
    # Return objective (we want to minimize -soft_min, so return val? No.)
    # _soft_min_neg returns val which is 1/beta log(sum exp(-beta x)).
    # This is an approximation of -min(x).
    # We want to MAXIMIZE min(x), so we MINIMIZE -min(x).
    # So the objective to minimize is val.
    
    return val, grad_centers.flatten()

def _initial_hex_grid(n_points):
    """Generate a hexagonal grid of points."""
    # Approximate hex grid
    # For N=200, ~14x14 grid.
    # Hex spacing: dx = s, dy = s * sqrt(3)/2
    # Area per point = s * s * sqrt(3)/2
    # Total area = 1.
    # s^2 * sqrt(3)/2 * 200 = 1 => s = sqrt(2/(200*sqrt(3)))
    
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate points
    points = []
    # Rows
    # Number of rows needed
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 2
    n_cols = int(1.0 / s) + 2
    
    # Start from bottom-left, offset
    # Center the grid
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height
    
    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            # Offset every other row
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            
            # Only keep points inside [-0.49, 0.49] to be safe
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have too many, trim randomly or by distance to center
    if len(points) > n_points:
        # Keep closest to center
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    # If too few, pad with random
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]

def construct():
    start_time = time.time()
    
    # 1. Initialize
    centers = _initial_hex_grid(N)
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
    centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    
    r_best = _compute_radius(centers)
    
    # 2. Gradient Descent with Annealing Beta
    # Start with small beta (smooth approximation), increase beta (sharper min)
    
    betas = [100, 500, 2000, 10000]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        # Run L-BFGS-B
        # We need to handle the case where the objective is not smooth enough or fails
        try:
            res = minimize(
                _objective_and_grad,
                centers.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            new_centers = res.x.reshape(-1, 2)
            # Clip to ensure validity
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
            
            r_new = _compute_radius(new_centers)
            if r_new > r_best:
                centers = new_centers
                r_best = r_new
        except Exception as e:
            # If optimization fails, keep current
            pass
            
    # 3. Inflation Phase
    # Try to increase the radius by optimizing the "inflation" directly.
    # This is harder. Instead, let's do a few more rounds of high-beta optimization.
    
    for beta in [50000, 100000]:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        try:
            res = minimize(
                _objective_and_grad,
                centers.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            new_centers = res.x.reshape(-1, 2)
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
            
            r_new = _compute_radius(new_centers)
            if r_new > r_best:
                centers = new_centers
                r_best = r_new
        except:
            pass
            
    # 4. Final local search / polish
    # Simple local moves: try to move the most constrained points.
    # Identify the tightest pairs and boundaries.
    
    # Let's do a few iterations of "move the point with the smallest margin".
    for _ in range(50):
        if time.time() - start_time > _DEADLINE - 1:
            break
            
        # Compute margins
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_margins = dists / 2.0
        
        bound_x = 0.5 - np.abs(centers[:, 0])
        bound_y = 0.5 - np.abs(centers[:, 1])
        bound_margins = np.minimum(bound_x, bound_y)
        
        # Find the point with the smallest minimum margin
        # min_margin_i = min(min_j pair_margins[i,j], bound_margins[i])
        
        min_pair_per_point = np.min(pair_margins, axis=1)
        min_margin_per_point = np.minimum(min_pair_per_point, bound_margins)
        
        # Find the point with the global minimum margin
        idx_min = np.argmin(min_margin_per_point)
        
        # Try to move this point slightly away from its constraint
        # Determine if it's a pair or boundary constraint
        if min_pair_per_point[idx_min] < bound_margins[idx_min]:
            # Pair constraint
            # Find the neighbor j
            j = np.argmin(pair_margins[idx_min])
            # Move away from j
            vec = centers[idx_min] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
                # Try moving in direction of vec
                step = 0.001
                new_c = centers.copy()
                new_c[idx_min] += vec * step
                # Clip
                new_c[idx_min, 0] = np.clip(new_c[idx_min, 0], -0.49, 0.49)
                new_c[idx_min, 1] = np.clip(new_c[idx_min, 1], -0.49, 0.49)
                
                r_new = _compute_radius(new_c)
                if r_new > r_best:
                    centers = new_c
                    r_best = r_new
        else:
            # Boundary constraint
            # Move towards center
            vec = -centers[idx_min]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
                step = 0.001
                new_c = centers.copy()
                new_c[idx_min] += vec * step
                new_c[idx_min, 0] = np.clip(new_c[idx_min, 0], -0.49, 0.49)
                new_c[idx_min, 1] = np.clip(new_c[idx_min, 1], -0.49, 0.49)
                
                r_new = _compute_radius(new_c)
                if r_new > r_best:
                    centers = new_c
                    r_best = r_best
    
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    return centers
