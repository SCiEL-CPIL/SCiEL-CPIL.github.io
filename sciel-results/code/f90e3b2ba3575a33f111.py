"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1
        # Height: (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
        
        # From hex geometry, nearest neighbor distance = s
        # So we need s >= 2*r
        
        # Let's solve: maximize r
        # s = 2*r (tight packing)
        # (n_cols - 1) * 2*r + 2*r <= 1  =>  2*r * n_cols <= 1  =>  r <= 1/(2*n_cols)
        # (n_rows - 1) * 2*r * sqrt(3)/2 + 2*r <= 1  =>  r * ((n_rows-1)*sqrt(3) + 2) <= 1
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        # Build the grid
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # Adjust: try to center the grid
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            # Now check if it fits
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _log_barrier_objective_and_grad(centers, min_pair_dist, min_wall_dist):
    """
    Objective: -log(min_pair_dist) - log(min_wall_dist)
    We want to maximize min_pair_dist and min_wall_dist.
    This is equivalent to minimizing the sum of negative logs.
    
    Gradient is computed using the chain rule on the active constraints.
    """
    n = centers.shape[0]
    obj_val = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise constraint: min |c_i - c_j|
    # Find the pair(s) with minimum distance
    # For efficiency in gradient, we can use the fact that at the optimum,
    # the gradient is a convex combination of the gradients of the active constraints.
    # Here we approximate by focusing on the closest pair.
    
    # To be more robust, we sum over all pairs that are "close" to the minimum.
    # However, for L-BFGS-B, a sparse gradient is better.
    # Let's find the global minimum pair.
    
    # We need to be careful with O(N^2) here.
    # Since N=200, N^2 = 40,000. This is fast in numpy.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Mask diagonal
    dists[np.arange(n), np.arange(n)] = np.inf
    
    min_idx_flat = np.argmin(dists)
    i_min, j_min = np.unravel_index(min_idx_flat, (n, n))
    
    # Ensure i < j for consistency, though symmetric
    if i_min > j_min:
        i_min, j_min = j_min, i_min
        
    d_min = dists[i_min, j_min]
    
    if d_min > 1e-12:
        # Contribution to objective: -log(d_min)
        obj_val += -np.log(d_min)
        
        # Gradient of -log(|c_i - c_j|) w.r.t c_i and c_j
        # d/dc_i (-log(||c_i - c_j||)) = -(c_i - c_j) / ||c_i - c_j||^2
        # d/dc_j (-log(||c_i - c_j||)) = (c_i - c_j) / ||c_i - c_j||^2
        
        vec = centers[i_min] - centers[j_min]
        d_sq = d_min * d_min
        factor = 1.0 / d_sq
        
        grad[i_min] += -vec * factor
        grad[j_min] += vec * factor
    
    # To handle multiple active constraints (degeneracies), we can add a small contribution from other close pairs
    # This makes the function smoother.
    # Let's add contributions from all pairs within 1.01 * d_min
    threshold = d_min * 1.01
    close_mask = (dists < threshold) & (dists > 1e-12)
    close_mask[np.arange(n), np.arange(n)] = False
    
    # Iterate over close pairs to accumulate gradient
    # This is O(N^2) but only for close pairs.
    for i in range(n):
        for j in range(i+1, n):
            if close_mask[i, j]:
                d = dists[i, j]
                vec = centers[i] - centers[j]
                d_sq = d * d
                # Weight by how close it is to the minimum?
                # Simple: just add it. This creates a "soft" minimum effect.
                # But we already added the exact min. Adding others might double count.
                # Better: Use a smooth min approximation?
                # Let's stick to the exact min for now to keep gradient clean,
                # but if there are ties, the gradient is undefined.
                # A common trick is to sum gradients of all active constraints with equal weight.
                # Let's try summing all pairs within a tight epsilon.
                pass # Skipping complex multi-active handling for speed, relying on L-BFGS-B robustness
    
    # 2. Wall constraint: min min_k (0.5 - |c_ik|)
    # Find the point(s) with minimum wall distance.
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    
    min_wall_idx = np.argmin(wall_dists)
    min_wall_val = wall_dists[min_wall_idx]
    
    if min_wall_val > 1e-12:
        obj_val += -np.log(min_wall_val)
        
        # Gradient of -log(wall_dist)
        # wall_dist = min(0.5 - x, x + 0.5) for 1D
        # If 0.5 - x < x + 0.5 => x > 0: dist = 0.5 - x, grad = -1
        # If 0.5 - x > x + 0.5 => x < 0: dist = x + 0.5, grad = 1
        
        for k in range(2):
            if centers[min_wall_idx, k] > 0:
                grad[min_wall_idx, k] += -1.0 / min_wall_val
            else:
                grad[min_wall_idx, k] += 1.0 / min_wall_val
    
    # Also add soft contribution from other close wall constraints
    wall_threshold = min_wall_val * 1.01
    close_wall_mask = (wall_dists < wall_threshold) & (wall_dists > 1e-12)
    
    for i in range(n):
        if close_wall_mask[i]:
            wd = wall_dists[i]
            for k in range(2):
                if centers[i, k] > 0:
                    grad[i, k] += -1.0 / wd
                else:
                    grad[i, k] += 1.0 / wd
                    
    # Note: The gradient of the minimum is not unique if there are ties.
    # Summing all active constraints is a valid subgradient.
    # However, the objective value should only count the minimum once to avoid scaling issues?
    # Actually, for optimization, minimizing the sum of -log of active constraints is a valid proxy
    # that pushes all active constraints up.
    
    # Wait, if I sum -log for multiple active constraints, the objective value changes scale.
    # It's better to keep the objective as -log(min) and use the subgradient.
    # But L-BFGS-B expects a consistent objective.
    # Let's refine: Objective = -log(min_pair) - log(min_wall).
    # Gradient = subgradient.
    
    # My current implementation sums -log for all "close" pairs in the gradient,
    # but only adds -log(min) to the objective value.
    # This mismatch can cause issues.
    
    # Let's fix: Objective value should match the gradient source.
    # Option A: Objective = sum_{active} -log(d_i). Gradient = sum_{active} grad_i.
    # This is a "soft" version of the min. It is smooth if we use a fixed set, but the set changes.
    # Option B: Objective = -log(min). Gradient = subgradient (e.g., average of active grads).
    
    # Let's implement Option A but with a smooth weighting based on distance to min.
    # This is getting complex. Let's stick to Option B:
    # Obj = -log(min_pair) - log(min_wall)
    # Grad = subgradient
    
    # Re-calculate Obj strictly as min
    obj_val = -np.log(d_min) - np.log(min_wall_val)
    
    # Re-calculate Grad as subgradient (sum of active grads is fine, or average)
    # Let's use the sum of active grads, normalized by count to keep scale similar?
    # No, just sum. It's a valid subgradient.
    
    grad_pair = np.zeros_like(centers)
    grad_wall = np.zeros_like(centers)
    
    # Pair active
    count_pair = 0
    for i in range(n):
        for j in range(i+1, n):
            if close_mask[i, j]:
                d = dists[i, j]
                vec = centers[i] - centers[j]
                d_sq = d * d
                factor = 1.0 / d_sq
                grad_pair[i] += -vec * factor
                grad_pair[j] += vec * factor
                count_pair += 1
                
    if count_pair > 0:
        # Average to keep gradient magnitude stable?
        # If we just sum, and there are 10 active pairs, the gradient is 10x larger.
        # This might be okay, but let's average.
        grad_pair /= count_pair

    # Wall active
    count_wall = 0
    for i in range(n):
        if close_wall_mask[i]:
            wd = wall_dists[i]
            for k in range(2):
                if centers[i, k] > 0:
                    grad_wall[i, k] += -1.0 / wd
                else:
                    grad_wall[i, k] += 1.0 / wd
            count_wall += 1
            
    if count_wall > 0:
        grad_wall /= count_wall

    grad = grad_pair + grad_wall
    
    return obj_val, grad

def _log_barrier_optimize(centers, max_iter=1000):
    """
    Optimize the log-barrier objective using L-BFGS-B.
    """
    centers = centers.copy()
    
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    x0 = centers.flatten()
    
    # We need to handle the case where min dist is 0 or very small
    # L-BFGS-B will handle it if we provide a valid gradient
    
    # Callback to check time
    def callback(xk):
        if _time_left() < 10.0:
            return True
        return False
        
    # L-BFGS-B parameters
    # ftol: precision of the minimization
    # gtol: gradient tolerance
    # maxiter: maximum number of iterations
    
    # We want high precision
    ftol = 1e-12
    gtol = 1e-8
    
    res = minimize(
        _log_barrier_objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'callback': callback, 'maxls': 20}
    )
    
    if res.success or res.x is not None:
        new_centers = res.x.reshape(N, 2)
        # Clip to ensure bounds are respected (L-BFGS-B should, but floating point)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        return new_centers
    else:
        return centers

def _local_refinement(centers, max_iter=500):
    """
    A simple local refinement step to polish the result.
    Uses a simple gradient ascent on the exact radius?
    Or just run L-BFGS-B again with tighter tolerances?
    """
    if _time_left() < 10.0:
        return centers
        
    # Run L-BFGS-B again with even tighter tolerances
    # This helps converge to the precise local optimum
    centers = _log_barrier_optimize(centers, max_iter=max_iter)
    return centers

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # Primary optimization: Log-barrier L-BFGS-B
    if _time_left() > 30.0:
        centers = _log_barrier_optimize(centers, max_iter=2000)
        r = _compute_radius(centers)
        
    # Secondary refinement: Tighter L-BFGS-B
    if _time_left() > 15.0:
        centers = _local_refinement(centers, max_iter=500)
        r = _compute_radius(centers)
        
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
