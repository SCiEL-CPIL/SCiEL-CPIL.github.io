"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_dense_hex(n):
    """
    Generate a dense hexagonal lattice and select n points optimally.
    Uses multiple spacing values and selects the best subset.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings
    spacings = np.linspace(0.068, 0.080, 15)
    
    for d in spacings:
        row_h = d * np.sqrt(3.0) / 2.0
        col_w = d
        
        # Generate all lattice points in [-0.5-d, 0.5+d]^2
        y_vals = np.arange(-0.5 - row_h, 0.5 + row_h + row_h, row_h)
        all_pts = []
        
        for j, y in enumerate(y_vals):
            if y < -0.5 - d or y > 0.5 + d:
                continue
            offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
            x_start = -0.5 - col_w + offset
            x_vals = np.arange(x_start, 0.5 + col_w + offset + col_w, col_w)
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    all_pts.append([x, y])
        
        if len(all_pts) < n:
            continue
        
        all_pts = np.array(all_pts)
        
        # Score each candidate point by how "central" it is
        # Points closer to boundary are more likely to limit radius
        dist_to_bnd = np.minimum(
            0.5 - np.abs(all_pts[:, 0]),
            0.5 - np.abs(all_pts[:, 1])
        )
        
        # Sort by distance to boundary (descending) - take most central points
        idx = np.argsort(dist_to_bnd)[::-1]
        selected = all_pts[idx[:n]]
        
        score = _calc_radius(selected)
        if score > best_score:
            best_score = score
            best_points = selected.copy()
    
    if best_points is None:
        # Fallback: regular grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    return best_points

def _qp_optimize(points, max_iter=5000, use_smooth=False):
    """
    Optimize point positions to maximize minimum distance using
    projected gradient descent with adaptive radius increase.
    
    Optionally uses a smooth log-sum-exp style approximation for the min-distance
    to get better gradients, combined with direct constraint handling.
    """
    n = len(points)
    r = _calc_radius(points)
    
    # Parameters
    lr_init = 0.006
    lr_min = 0.00003
    beta = 0.9995  # Slower decay for more consistent learning
    r_step = 0.0001
    smooth_param = 0.003  # Smoothing parameter for log-sum-exp
    
    points = points.copy()
    
    for it in range(max_iter):
        lr = max(lr_init * (beta ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # --- Pairwise gradient contribution ---
        if use_smooth:
            # Soft log-sum-exp approach: use all pairs with smooth weights
            # weight_ij = exp(-(dists_ij - 2r)^2 / (2*sigma^2)) for dists < 2r + margin
            sigma = smooth_param
            margin = 4 * sigma
            mask = dists < (2.0 * r + margin)
            safe_dists = np.where(dists > 1e-10, dists, 1e-10)
            
            # Gaussian-like weights centered at 2r
            viol = np.where(mask, 2.0 * r - dists, 0.0)
            weights = np.zeros_like(dists)
            weights[mask] = np.exp(-0.5 * (viol[mask] / sigma) ** 2)
            
            # Normalize weights
            row_sums = np.sum(weights, axis=1)
            row_sums[row_sums == 0] = 1.0
            weights /= row_sums[:, np.newaxis]
            
            # Gradient: push apart based on weighted differences
            weighted_sum = weights @ points
            pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
            pair_grad /= np.maximum(row_sums[:, np.newaxis], 1e-10)
        else:
            # Hard constraint approach: only push violating pairs
            mask = dists < 2.0 * r
            safe_dists = np.where(dists > 1e-10, dists, 1e-10)
            
            # Violation amount
            viol = np.where(mask, 2.0 * r - dists, 0.0)
            
            # Weight for each pair: proportional to violation
            weights = np.zeros_like(dists)
            weights[mask] = viol[mask] / safe_dists[mask]
            
            # Gradient from pairwise constraints
            row_sums = np.sum(weights, axis=1)
            weighted_sum = weights @ points
            pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # --- Boundary gradient contribution ---
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Active boundary constraint per point
        active_x = bx <= by
        
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        # Gradient: push away from boundary with smooth transition
        # Use tanh-like smooth sign
        smooth_sign_x = np.tanh(20.0 * points[:, 0])
        smooth_sign_y = np.tanh(20.0 * points[:, 1])
        
        grad_x = viol_x * smooth_sign_x
        grad_y = viol_y * smooth_sign_y
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 0:
            total_grad = total_grad / grad_norm * min(grad_norm, 0.1)
        
        # Update positions
        points = points - lr * total_grad
        
        # Project back into square with margin
        margin = 0.5 - r
        if margin > 0:
            points[:, 0] = np.clip(points[:, 0], -margin, margin)
            points[:, 1] = np.clip(points[:, 1], -margin, margin)
        else:
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check current radius
        current_r = _calc_radius(points)
        
        # Adaptive radius update with hysteresis
        if current_r >= r:
            # Can increase r slightly, but conservatively
            r = min(current_r + r_step * 0.3, current_r + 1e-7)
        else:
            # We lost some radius; reduce r to recover
            r = max(current_r, r - r_step * 0.5)
    
    return points

def _multi_start_optimize(initial_points, n_starts=3, max_iter_per_start=3000):
    """
    Try multiple optimization starts with slight perturbations and keep the best.
    """
    best_points = initial_points.copy()
    best_score = _calc_radius(best_points)
    
    for start in range(n_starts):
        # Create perturbed version
        if start == 0:
            pts = initial_points.copy()
        else:
            # Small random perturbation
            np.random.seed(start * 42 + 7)
            perturbation = np.random.randn(len(initial_points), 2) * 0.002
            pts = initial_points + perturbation
            pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
            pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        # Optimize with smooth gradient
        optimized = _qp_optimize(pts, max_iter=max_iter_per_start, use_smooth=True)
        
        score = _calc_radius(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    return best_points

def _final_polish(points, max_iter=2000):
    """
    Final high-precision polish with very small steps.
    """
    n = len(points)
    r = _calc_radius(points)
    
    lr_init = 0.001
    lr_min = 0.00001
    beta = 0.999
    
    points = points.copy()
    
    for it in range(max_iter):
        lr = max(lr_init * (beta ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs close to the minimum
        min_dist = np.min(dists)
        threshold = min_dist * 1.05
        
        mask = dists < threshold
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        # Equal weight for all near-minimum pairs
        weights = np.zeros_like(dists)
        weights[mask] = 1.0 / safe_dists[mask]
        
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # Boundary handling
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        active_x = bx <= by
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        smooth_sign_x = np.tanh(50.0 * points[:, 0])
        smooth_sign_y = np.tanh(50.0 * points[:, 1])
        
        grad_x = viol_x * smooth_sign_x
        grad_y = viol_y * smooth_sign_y
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # Very small step
        points = points - lr * total_grad
        points[:, 0] = np.clip(points[:, 0], -0.5 + r, 0.5 - r)
        points[:, 1] = np.clip(points[:, 1], -0.5 + r, 0.5 - r)
        
        current_r = _calc_radius(points)
        if current_r >= r:
            r = current_r
        else:
            r = max(current_r, r - 0.00001)
    
    return points

def construct():
    # Step 1: Generate initial hexagonal lattice
    points = _generate_dense_hex(N)
    
    # Step 2: Multi-start optimization with smooth gradients
    points = _multi_start_optimize(points, n_starts=3, max_iter_per_start=3000)
    
    # Step 3: Final high-precision polish
    points = _final_polish(points, max_iter=2000)
    
    return points
# EVOLVE-BLOCK-END
