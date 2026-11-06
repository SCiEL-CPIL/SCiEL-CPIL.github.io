import numpy as np
import time

N = 200
DEADLINE = 140.0  # Safety margin before 150s limit


def _compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 0.5):
        return 0.0
    if np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 0.5):
        return 0.0

    # Center-center distances (vectorized)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_center_dist = np.sqrt(np.min(dist_sq))

    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))

    r = min(min_center_dist, min_wall_dist) / 2.0
    return r


def _generate_hex_grid(n):
    """Generate a hexagonal grid of points inside the unit square."""
    # For N=200, use a grid that fits well
    # Hex packing: rows of alternating offset
    # Try to fit ~15 cols x ~14 rows = 210, then trim
    cols = 15
    rows = 14
    s = 1.0 / (cols + 0.5)  # spacing
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Filter points inside the square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
           (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > n:
        # Select n points greedily by max min-distance
        selected = [0]
        remaining = list(range(1, len(points)))
        while len(selected) < n and remaining:
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                min_d = min(np.linalg.norm(points[idx] - points[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        points = points[selected]
    elif len(points) < n:
        missing = n - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (missing, 2))
        points = np.vstack([points, random_pts])
    
    return points


def _vectorized_optimize(points, r_target, n_iters, start_time, deadline, 
                          lr=0.001, decay=0.9995, hinge_width=0.005):
    """
    Vectorized gradient descent on smooth hinge loss.
    Objective: minimize sum of hinge(2*r_target - dist_ij) for all pairs + wall penalties
    This pushes all pairs to be at least 2*r_target apart.
    """
    points = points.copy().astype(np.float64)
    lr_current = lr
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute all pairwise differences and distances (vectorized)
        # diff[i,j] = points[i] - points[j], shape (N, N, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 2)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)
        np.fill_diagonal(dist_sq, 1e10)
        
        # Avoid division by zero
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        # Hinge loss: only pairs closer than 2*r_target contribute
        target_dist = 2.0 * r_target
        violation = target_dist - dist  # positive when too close
        # Smooth hinge: use softplus-like for differentiability
        # hinge = max(0, violation) approximated smoothly
        # Use: 0.5 * (violation + sqrt(violation^2 + eps^2)) - 0.5*eps
        eps = hinge_width
        smooth_violation = 0.5 * (violation + np.sqrt(violation**2 + eps**2)) - 0.5 * eps
        # Only positive violations matter
        smooth_violation = np.maximum(smooth_violation, 0.0)
        
        # Gradient of smooth hinge w.r.t. dist:
        # d(smooth_violation)/d(dist) = -violation / sqrt(violation^2 + eps^2) (when violation > 0)
        # d(dist)/d(points[i]) = (points[i] - points[j]) / dist
        # So force on i from j: -d(loss)/d(points[i]) = violation/dist * (points[i]-points[j])/dist
        # = violation * diff / dist^2
        
        # Compute force direction
        # force[i,j] = violation[i,j] * diff[i,j] / dist_sq[i,j]  (for i != j)
        with np.errstate(divide='ignore', invalid='ignore'):
            force_weight = smooth_violation / np.maximum(dist_sq, 1e-10)
        
        # Set diagonal to 0
        np.fill_diagonal(force_weight, 0.0)
        
        # Force on point i = sum over j of force_weight[i,j] * diff[i,j]
        # diff[i,j] = points[i] - points[j]
        # So force[i] = sum_j force_weight[i,j] * (points[i] - points[j])
        # = points[i] * sum_j force_weight[i,j] - sum_j force_weight[i,j] * points[j]
        
        fw_row_sum = np.sum(force_weight, axis=1)  # (N,)
        # Weighted sum of points: (N, 2)
        weighted_pts = force_weight @ points  # (N, 2)
        
        forces = points * fw_row_sum[:, np.newaxis] - weighted_pts
        
        # Wall forces: push away from walls
        # Left wall: if x < r_target, push right
        # Right wall: if x > 0.5 - r_target, push left
        wall_margin = r_target
        for axis in range(2):
            below = points[:, axis] < (-0.5 + wall_margin)
            above = points[:, axis] > (0.5 - wall_margin)
            forces[below, axis] += (wall_margin - 0.5 - points[below, axis]) * 2.0
            forces[above, axis] -= (points[above, axis] - (0.5 - wall_margin)) * 2.0
        
        # Apply gradient step (maximize radius => minimize loss => subtract gradient)
        # We're minimizing the hinge loss, so we move in negative gradient direction
        # But our "forces" above push points apart (increase distances), which is what we want
        # Actually, let's reconsider: we want to INCREASE minimum distance
        # The hinge loss penalizes small distances, so minimizing it increases distances
        # Gradient descent: points -= lr * d(loss)/d(points)
        # d(loss)/d(points[i]) = -forces[i] (from our derivation above, forces push apart)
        # So points += lr * forces[i]
        points += lr_current * forces
        
        # Clip to stay inside square (with small margin)
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
        
        lr_current *= decay
    
    return points


def _polish_vectorized(points, r_target, n_iters, start_time, deadline,
                        lr=0.0005, decay=0.9998, hinge_width=0.002):
    """Fine polishing with smaller step size and tighter hinge."""
    return _vectorized_optimize(points, r_target, n_iters, start_time, deadline,
                                lr=lr, decay=decay, hinge_width=hinge_width)


def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Generate initial hex grid
    init_points = _generate_hex_grid(N)
    init_points = np.clip(init_points, -0.48, 0.48)
    
    best_points = init_points.copy()
    best_score = _compute_radius(init_points)
    
    # Multi-scale optimization: gradually increase target radius
    # Known best is ~0.0366, aim for that range
    
    # Estimate initial feasible radius
    r_start = 0.02
    
    # Create a sequence of target radii to try
    r_targets = []
    r = r_start
    while r < 0.0365:
        r_targets.append(r)
        r *= 1.04  # 4% increase per step
    r_targets.append(0.0365)
    r_targets.append(0.0366)
    r_targets.append(0.0367)
    
    for r_target in r_targets:
        if time.time() > DEADLINE - 40:
            break
        
        # Run vectorized optimization
        points = _vectorized_optimize(best_points, r_target, 
                                       n_iters=2000,
                                       start_time=start_time,
                                       deadline=DEADLINE,
                                       lr=0.002,
                                       decay=0.9997,
                                       hinge_width=0.005)
        
        actual_r = _compute_radius(points)
        
        if actual_r > best_score:
            best_score = actual_r
            best_points = points.copy()
        
        # If we achieved close to target, polish
        if actual_r >= r_target * 0.98 and time.time() < DEADLINE - 20:
            polished = _polish_vectorized(best_points, best_score,
                                           n_iters=1000,
                                           start_time=start_time,
                                           deadline=DEADLINE,
                                           lr=0.0005,
                                           decay=0.9998,
                                           hinge_width=0.002)
            polished_r = _compute_radius(polished)
            if polished_r > best_score:
                best_score = polished_r
                best_points = polished.copy()
    
    # Final polish with remaining time
    if time.time() < DEADLINE - 10:
        remaining_time = DEADLINE - 10 - time.time()
        n_final_iters = int(min(5000, max(500, remaining_time * 50)))
        final_points = _vectorized_optimize(best_points, best_score,
                                             n_iters=n_final_iters,
                                             start_time=start_time,
                                             deadline=DEADLINE,
                                             lr=0.0003,
                                             decay=0.9999,
                                             hinge_width=0.001)
        final_score = _compute_radius(final_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Ensure all points are strictly inside
    best_points[:, 0] = np.clip(best_points[:, 0], -0.49999, 0.49999)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.49999, 0.49999)
    
    return best_points
