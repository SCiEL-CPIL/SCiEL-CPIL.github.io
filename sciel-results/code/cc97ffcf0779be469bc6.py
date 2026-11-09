"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the exact minimum radius."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Boundary distance
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_radius(positions, t):
    """
    Compute the soft-minimum of the radius constraints.
    Returns a smooth approximation of min(boundary_dists, pair_dists/2).
    """
    n = len(positions)
    
    # Boundary distances
    boundary_dists = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    pair_radii = np.sqrt(dist_sq) / 2.0
    
    # Combine all radii
    # We only need upper triangle for pairs
    iu = np.triu_indices(n, k=1)
    pair_radii_flat = pair_radii[iu]
    
    all_radii = np.concatenate([boundary_dists, pair_radii_flat])
    
    # Soft min: -t * log(sum(exp(-r_i / t)))
    # For stability, shift by max
    max_r = np.max(all_radii)
    neg_scaled = -(all_radii - max_r) / t
    # If t is very small, this might underflow for large negative values, but exp(0)=1 for max
    # Actually, if all_radii are around 0.03, and t=0.1, exp(-0.03/0.1) is fine.
    # If t is very small, e.g. 0.01, exp(-3) is fine.
    
    # Use logsumexp for stability
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -all_radii / t
    # We want -t * logsumexp(-all_radii / t)
    
    x = -all_radii / t
    max_x = np.max(x)
    if not np.isfinite(max_x):
        return 0.0
        
    log_sum = np.log(np.sum(np.exp(x - max_x))) + max_x
    soft_min_val = -t * log_sum
    
    return soft_min_val

def _generate_hexagonal_grid(n_circles):
    """Generate a hexagonal grid centered at origin."""
    points = []
    # Estimate number of rows needed
    # Area per circle ~ 2 * r^2 * sqrt(3) / 2 ? No, hex packing density 0.9069
    # r ~ 0.036. Area ~ 0.004. 200 circles ~ 0.8 area.
    # Side ~ 0.9.
    # Rows ~ 0.9 / (r*sqrt(3)) ~ 0.9 / 0.06 ~ 15 rows.
    max_rows = 20
    d = 0.05 # initial spacing guess, will be scaled
    
    for row in range(-max_rows, max_rows + 1):
        offset = (row % 2) * 0.5 * d
        for col in range(-max_rows, max_rows + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            points.append([x, y])
            
    points = np.array(points)
    
    # Sort by distance to center to get the most central points
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx]
    
    # Scale to fit in [-0.5, 0.5] with some margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points = points * (0.45 / max_coord) # Keep away from edges initially
        
    return points

def _optimize_with_soft_min(positions, bounds, t_start=50.0, t_end=0.5, n_anneal_steps=10, max_iter_per_step=200):
    """
    Optimize positions using L-BFGS-B on the negative soft-min radius.
    Anneals the temperature t.
    """
    current_pos = positions.copy()
    n = len(current_pos)
    
    for i in range(n_anneal_steps):
        # Anneal temperature
        t_val = t_start * (t_end / t_start) ** (i / (n_anneal_steps - 1)) if n_anneal_steps > 1 else t_start
        
        def _objective(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            # Soft penalty for out of bounds (though bounds are enforced by L-BFGS-B)
            # The soft-min itself penalizes boundary violations by reducing the radius
            # But if a point goes far out, boundary dist is negative, soft-min is negative.
            # We minimize -soft_min. So negative soft-min -> positive objective (bad).
            # So it naturally repels from boundaries.
            return -_soft_min_radius(pos, t_val)
            
        try:
            res = minimize(
                _objective,
                current_pos.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_step, 'maxfun': max_iter_per_step * 5, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            current_pos = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    return current_pos

def _local_push_optimize(positions, max_iters=500, tol=1e-8):
    """
    Local search: Find the closest pair (or boundary violation) and push it apart.
    """
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iters):
        # Calculate current score components
        min_boundary = np.min(np.minimum(
            0.5 - np.abs(positions[:, 0]),
            0.5 - np.abs(positions[:, 1])
        ))
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        min_pair_radius = min_pair_dist / 2.0
        
        current_r = min(min_boundary, min_pair_radius)
        
        # Identify the constraint that is limiting the radius
        if min_boundary < min_pair_radius:
            # Boundary is limiting
            # Find point(s) closest to boundary
            # dist_to_boundary for each point
            d_b = np.minimum(
                0.5 - np.abs(positions[:, 0]),
                0.5 - np.abs(positions[:, 1])
            )
            min_idx = np.argmin(d_b)
            p = positions[min_idx]
            
            # Determine direction to push inward
            # If x is closer to boundary than y
            dx = 0.5 - abs(p[0])
            dy = 0.5 - abs(p[1])
            
            if dx < dy:
                # Push in x
                if p[0] > 0:
                    positions[min_idx, 0] -= 0.001
                else:
                    positions[min_idx, 0] += 0.001
            else:
                # Push in y
                if p[1] > 0:
                    positions[min_idx, 1] -= 0.001
                else:
                    positions[min_idx, 1] += 0.001
            
            # Clip to bounds
            positions = np.clip(positions, -0.5, 0.5)
            
        else:
            # Pair is limiting
            min_idx = np.argmin(dist)
            i, j = np.unravel_index(min_idx, (n, n))
            if i >= j:
                i, j = j, i
                
            p_i = positions[i]
            p_j = positions[j]
            
            direction = p_i - p_j
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                # Random direction
                direction = np.array([1.0, 0.0])
                norm = 1.0
            else:
                direction = direction / norm
                
            # Push apart
            push_amount = 0.001
            positions[i] += direction * push_amount
            positions[j] -= direction * push_amount
            
            # Clip to bounds
            positions = np.clip(positions, -0.5, 0.5)
            
            # If clipping changed the distance significantly, we might have made it worse.
            # But for local push, this is usually fine.
            
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_positions = None
    
    # 1. Initialization
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
        
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # 2. Phase 1: Soft-min optimization (Annealed)
    if time.time() < deadline - 40:
        try:
            opt_pos = _optimize_with_soft_min(best_positions, bounds, t_start=100.0, t_end=1.0, n_anneal_steps=8, max_iter_per_step=100)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
            
    # 3. Phase 2: Local Push Refinement
    if time.time() < deadline - 30:
        try:
            pushed_pos = _local_push_optimize(best_positions, max_iters=1000)
            push_score = _score(pushed_pos)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed_pos.copy()
        except Exception:
            pass
            
    # 4. Phase 3: Final Soft-min Polish (Low T)
    if time.time() < deadline - 20:
        try:
            # Use a very low temperature for precise adjustment
            polish_pos = _optimize_with_soft_min(best_positions, bounds, t_start=1.0, t_end=0.1, n_anneal_steps=5, max_iter_per_step=100)
            polish_score = _score(polish_pos)
            if polish_score > best_score:
                best_score = polish_score
                best_positions = polish_pos.copy()
        except Exception:
            pass
            
    # 5. Phase 4: Final Local Push
    if time.time() < deadline - 10:
        try:
            final_push = _local_push_optimize(best_positions, max_iters=500)
            final_score = _score(final_push)
            if final_score > best_score:
                best_score = final_score
                best_positions = final_push.copy()
        except Exception:
            pass

    # Ensure valid output
    if best_positions is None:
        best_positions = _generate_hexagonal_grid(N)
        
    # Final sanity check: clip to bounds
    best_positions = np.clip(best_positions, -0.5, 0.5)
    
    return best_positions

# EVOLVE-BLOCK-END
