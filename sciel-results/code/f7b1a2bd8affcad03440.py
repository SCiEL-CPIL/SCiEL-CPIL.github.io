import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    """
    Find 50 circle centers in [-0.5, 0.5]^2 that maximize the minimum radius.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def compute_radius(centers):
        """Compute the maximum common radius for a given set of centers."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        min_boundary = 0.5 - np.max(np.abs(centers), axis=1).max()
        
        return min(min_pairwise / 2, min_boundary)

    def smooth_objective(centers_flat, beta=100):
        """
        Smooth approximation of the negative minimum radius.
        """
        c = centers_flat.reshape(N, 2)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        mask = ~np.eye(N, dtype=bool)
        dists_masked = dists[mask]
        
        # Soft min of pairwise distances / 2
        min_dists_soft = -1.0 / beta * np.log(np.sum(np.exp(-beta * dists_masked)))
        
        # Boundary soft min
        max_abs = np.max(np.abs(c), axis=1)
        boundary_clearances = 0.5 - max_abs
        min_boundary_soft = -1.0 / beta * np.log(np.sum(np.exp(-beta * boundary_clearances)))
        
        # Overall soft min
        r_soft = min(min_dists_soft / 2, min_boundary_soft)
        
        # Penalty for boundary violations
        boundary_violation = np.maximum(0, max_abs - 0.5)
        penalty = np.sum(boundary_violation**2) * 1000
        
        return -r_soft + penalty

    def vectorized_repulsion_search(centers, r_target, max_iters=2000, step_size=0.01):
        """
        Vectorized force-directed search to resolve overlaps for a given r_target.
        """
        c = centers.copy()
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Forces from pairwise repulsion
            # Only consider pairs that are too close
            mask = dists < 2 * r_target
            mask &= ~np.eye(N, dtype=bool)
            
            # Force magnitude: proportional to (2*r_target - dist)
            # Direction: normalized difference
            # We want to push apart, so force on i is -dir * mag, on j is +dir * mag
            # Let's compute the sum of forces for each i
            
            forces = np.zeros_like(c)
            
            # Vectorized calculation of forces is tricky with masking, 
            # but we can do it by iterating over active pairs if few, or full matrix.
            # For N=50, full matrix is small.
            
            # d_ij = dists[i,j]
            # dir_ij = (c[i] - c[j]) / d_ij
            # force_ij = k * (2*r_target - d_ij) * dir_ij
            # F[i] -= force_ij
            # F[j] += force_ij
            
            # To vectorize:
            # Create a matrix of forces F_mat where F_mat[i,j] is force on i due to j
            # F_mat[i,j] = (c[i] - c[j]) / dists[i,j] * (2*r_target - dists[i,j]) * mask[i,j]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                dir_mat = diff / dists[:, :, np.newaxis]
                mag_mat = np.where(mask, (2 * r_target - dists), 0)
                force_mat = dir_mat * mag_mat[:, :, np.newaxis]
            
            # Sum forces for each i
            # F[i] = sum_j force_mat[i,j]
            forces = np.sum(force_mat, axis=1)
            
            # Boundary forces
            # Push away from boundaries
            # If c[i,k] > 0.5 - r_target, push left
            # If c[i,k] < -0.5 + r_target, push right
            boundary_forces = np.zeros_like(c)
            for k in range(2):
                # Right boundary
                mask_right = c[:, k] > (0.5 - r_target)
                boundary_forces[mask_right, k] -= (c[mask_right, k] - (0.5 - r_target))
                
                # Left boundary
                mask_left = c[:, k] < (-0.5 + r_target)
                boundary_forces[mask_left, k] += ((-0.5 + r_target) - c[mask_left, k])
            
            total_forces = forces + boundary_forces
            
            # Update positions
            c += step_size * total_forces
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check if converged (no more overlaps)
            min_dist = np.min(dists[mask]) if np.any(mask) else np.inf
            if min_dist >= 2 * r_target:
                break
        
        return c

    # Initial configurations
    initializers = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initializers.append(grid[:N].copy())
    
    # 2. Hexagonal
    n_rows = 7
    pts = []
    for row in range(n_rows):
        y = -0.5 + (row + 0.5) / n_rows
        offset = (row % 2) * 0.05
        n_cols = 8
        for col in range(n_cols):
            x = -0.5 + (col + 0.5) / n_cols + offset
            pts.append([x, y])
    hex_grid = np.array(pts[:N])
    hex_grid = np.clip(hex_grid, -0.49, 0.49)
    initializers.append(hex_grid)
    
    # 3. Random
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 2))
    initializers.append(random_init)
    
    best_centers = None
    best_val = -np.inf
    
    for idx, init in enumerate(initializers):
        if time.time() > deadline - 30:
            break
        
        # Start with a small radius and increase
        r_current = compute_radius(init)
        r_target = r_current + 0.001
        
        c_current = init.copy()
        
        while time.time() < deadline - 20 and r_target < 0.1:
            # Try to resolve overlaps for r_target
            c_new = vectorized_repulsion_search(c_current, r_target, max_iters=500, step_size=0.02)
            
            # Check if we succeeded
            r_new = compute_radius(c_new)
            
            if r_new >= r_target - 1e-6:
                c_current = c_new
                r_current = r_new
                r_target += 0.0005
            else:
                # Failed to reach r_target, maybe stuck
                # Add some noise and retry
                c_current = c_new + np.random.normal(0, 0.01, size=c_new.shape)
                c_current = np.clip(c_current, -0.49, 0.49)
                # Reduce target slightly to recover
                r_target = max(r_current + 0.0001, r_target - 0.001)
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    smooth_objective,
                    c_current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 100000, 'gtol': 1e-6},
                    bounds=[(-0.5, 0.5)] * (2 * N)
                )
                c_opt = result.x.reshape(N, 2)
                c_opt = np.clip(c_opt, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = compute_radius(c_opt)
                
                if r_opt > best_val:
                    best_val = r_opt
                    best_centers = c_opt.copy()
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = initializers[0]
    
    return best_centers
