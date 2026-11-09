"""Circle packing in perimeter-4 rectangle using vectorized soft-body relaxation."""
N = 21
import numpy as np
import time
import random
from scipy.optimize import minimize

def _validate(x, y, r, w, h, tol=1e-7):
    """Check if a packing is valid."""
    if w <= 0 or h <= 0 or abs(w + h - 2.0) > 1e-6:
        return False
    for i in range(N):
        if r[i] < tol:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _soft_body_relax(x, y, r, w, h, max_iters=2000, tol=1e-8):
    """
    Physics-based relaxation using vectorized forces.
    1. Repulsive forces between circles (Lennard-Jones-like or simple repulsion).
    2. Boundary constraints handled via projection or penalty.
    3. Radius expansion: try to increase radii while maintaining non-overlap.
    
    We use a simple iterative scheme:
    - Compute forces based on overlaps.
    - Update positions.
    - Attempt to increase radii slightly if no overlaps.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    r = np.array(r, dtype=float)
    
    # Initial step size
    step = 0.01
    min_step = 1e-6
    no_improve_count = 0
    best_score = np.sum(r)
    
    for it in range(max_iters):
        # 1. Compute pairwise distances
        # Vectorized distance calculation
        # dx[i,j] = x[i] - x[j]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0) 
        dist = np.sqrt(dist_sq)
        
        # 2. Compute forces
        # Force is proportional to overlap depth for overlapping pairs.
        # We want to push circles apart.
        # Overlap depth d_ij = (r_i + r_j) - dist_ij
        # Force vector F_ij on i from j: (d_ij / dist_ij) * (dx_ij, dy_ij)
        
        # Mask for pairs where dist < r_i + r_j
        min_dist = r[:, None] + r[None, :]
        overlap_depth = np.maximum(min_dist - dist, 0.0)
        
        # Normalize direction
        # dir_ij = (dx_ij, dy_ij) / dist_ij
        # Note: dx[i,j] is x[i]-x[j]. Force on i should be in direction of x[i]-x[j] if overlapping.
        
        # Safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_x = dx / dist
            dir_y = dy / dist
            
        # Force on i is sum over j of overlap_depth[i,j] * dir[i,j]
        # This pushes i away from j.
        force_x = np.sum(overlap_depth * dir_x, axis=1)
        force_y = np.sum(overlap_depth * dir_y, axis=1)
        
        # 3. Boundary forces
        # If x[i] < r[i], push right. If x[i] > w - r[i], push left.
        # Similarly for y.
        bound_force_x = np.zeros(N)
        bound_force_y = np.zeros(N)
        
        # Left boundary
        mask_left = x < r
        bound_force_x[mask_left] += (r[mask_left] - x[mask_left])
        
        # Right boundary
        mask_right = x > w - r
        bound_force_x[mask_right] -= (x[mask_right] - (w - r[mask_right]))
        
        # Bottom boundary
        mask_bottom = y < r
        bound_force_y[mask_bottom] += (r[mask_bottom] - y[mask_bottom])
        
        # Top boundary
        mask_top = y > h - r
        bound_force_y[mask_top] -= (y[mask_top] - (h - r[mask_top]))
        
        # Total force
        total_fx = force_x + bound_force_x
        total_fy = force_y + bound_force_y
        
        # 4. Update positions
        # Simple gradient descent on potential energy
        x_new = x + step * total_fx
        y_new = y + step * total_fy
        
        # 5. Project positions back into valid range [r, w-r] and [r, h-r]
        # This ensures boundary constraints are strictly satisfied after update.
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)
        
        # 6. Try to expand radii
        # Check if new positions are non-overlapping
        # If yes, we can try to increase radii.
        # How much? We can find the max possible increase based on current gaps.
        # Or simply try a small increase and check validity.
        
        # Let's calculate the minimum gap to neighbors and boundaries.
        # gap_ij = dist_ij - (r_i + r_j)
        # We want to increase r_i by dr such that it doesn't cause overlap.
        # This is complex to do exactly for all i simultaneously.
        # Simple heuristic: if no overlaps in new positions, increase all r by a small amount.
        
        # Check overlaps in new positions
        dx_new = x_new[:, None] - x_new[None, :]
        dy_new = y_new[:, None] - y_new[None, :]
        dist_new_sq = dx_new**2 + dy_new**2
        np.fill_diagonal(dist_new_sq, 1.0)
        dist_new = np.sqrt(dist_new_sq)
        
        min_dist_new = r[:, None] + r[None, :]
        overlaps_new = np.sum(dist_new < min_dist_new - 1e-9)
        
        if overlaps_new == 0:
            # No overlaps, try to increase radii
            # Limit increase by distance to boundaries
            max_r_by_bound = np.minimum(np.minimum(x_new, w - x_new), np.minimum(y_new, h - y_new))
            # Also limit by neighbor distances
            # For each i, max increase is min over j of (dist_ij - r_i - r_j) / 2 ?
            # Actually, if we increase r_i by dr, it affects dist to all j.
            # Let's just try a small uniform increase.
            dr = 0.001
            r_new = r + dr
            
            # Check if this increase causes overlaps
            min_dist_r_new = r_new[:, None] + r_new[None, :]
            overlaps_r_new = np.sum(dist_new < min_dist_r_new - 1e-9)
            
            if overlaps_r_new == 0:
                # Also check boundaries
                valid_bounds = np.all(x_new >= r_new - 1e-9) and np.all(x_new <= w - r_new + 1e-9) and \
                               np.all(y_new >= r_new - 1e-9) and np.all(y_new <= h - r_new + 1e-9)
                if valid_bounds:
                    r = r_new
                    x = x_new
                    y = y_new
                    no_improve_count = 0
                else:
                    # Boundary violation, just accept position update
                    x = x_new
                    y = y_new
                    no_improve_count += 1
            else:
                # Overlap caused by radius increase, accept position update only
                x = x_new
                y = y_new
                no_improve_count += 1
        else:
            # Still overlaps, accept position update
            x = x_new
            y = y_new
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 10:
            step *= 0.9
            if step < min_step:
                break
        
        # Check convergence
        current_score = np.sum(r)
        if abs(current_score - best_score) < 1e-6 and no_improve_count > 50:
            break
        best_score = current_score

    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_solution = None
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        # w + h = 2
        # Sample w from [0.6, 1.4]
        w = random.uniform(0.6, 1.4)
        h = 2.0 - w
        
        # Initialize circles
        # Use a grid-like initialization with jitter
        rows = 5
        cols = 5
        # 25 spots, pick 21
        points = [(i, j) for i in range(cols) for j in range(rows)]
        random.shuffle(points)
        selected = points[:N]
        
        # Grid centers
        x = np.array([(i + 0.5) * (w / cols) for i, j in selected])
        y = np.array([(j + 0.5) * (h / rows) for i, j in selected])
        
        # Add jitter
        jitter = 0.02
        x += np.random.uniform(-jitter, jitter, N)
        y += np.random.uniform(-jitter, jitter, N)
        
        # Ensure within bounds
        # Initial radii
        cell_w = w / cols
        cell_h = h / rows
        r = np.full(N, min(cell_w, cell_h) * 0.4)
        
        # Ensure r fits in bounds
        r = np.minimum(r, np.minimum(x, w - x))
        r = np.minimum(r, np.minimum(y, h - y))
        r = np.maximum(r, 0.01)
        
        # Run soft-body relaxation
        x_rel, y_rel, r_rel = _soft_body_relax(x, y, r, w, h, max_iters=500)
        
        # Verify validity
        if not _validate(x_rel, y_rel, r_rel, w, h):
            # Try to fix by shrinking radii slightly if needed?
            # Or just skip
            continue
            
        score = np.sum(r_rel)
        if score > best_score:
            best_score = score
            best_solution = np.column_stack((x_rel, y_rel, r_rel))
            
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
