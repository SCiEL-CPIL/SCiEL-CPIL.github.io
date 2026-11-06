import numpy as np
import time
import math

N = 100

def _energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient for point i: grad_i = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # points shape (N, 3)
    # Compute pairwise differences
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Compute squared distances
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i < j of -log(d_ij)
    # We can sum over all i != j and divide by 2
    # log(d_ij) is symmetric
    # E = 0.5 * sum_{i != j} -log(d_ij)
    # Mask diagonal to ignore self terms
    mask = ~np.eye(N, dtype=bool)
    
    log_d = np.zeros_like(dist_sq)
    log_d[mask] = -np.log(dist[mask])
    
    energy = 0.5 * np.sum(log_d)
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_sq[i, j]
    # term[i, j] = (x_i - x_j) / dist_sq[i, j]
    # We want sum_j term[i, j]
    
    # Initialize gradient
    grad = np.zeros_like(points)
    
    # Compute terms
    # diff is (N, N, 3)
    # dist_sq is (N, N)
    # We divide each vector diff[i, j, :] by dist_sq[i, j]
    # Note: dist_sq is 1 on diagonal, so diff[i,i]/1 = 0, which is correct.
    
    # To avoid division by zero issues if any distance is 0 (shouldn't happen with mask/fill_diagonal logic if points distinct)
    # But dist_sq is filled with 1 on diagonal.
    
    # inv_dist_sq = 1.0 / dist_sq
    # term = diff * inv_dist_sq[:, :, np.newaxis]
    # grad = -np.sum(term, axis=1)
    
    # Let's do it carefully to avoid memory spikes if N was huge, but N=100 is small.
    inv_dist_sq = 1.0 / dist_sq
    # Zero out diagonal just in case, though diff is 0 there
    inv_dist_sq[mask] = inv_dist_sq[mask]
    # Diagonal inv_dist_sq is 1, diff is 0, so term is 0.
    
    terms = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j (axis 1)
    grad = -np.sum(terms, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    current_energy, _ = _energy_and_grad(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # 2. Optimization Loop: Projected Gradient Descent
    # We will use a backtracking line search to determine step size.
    # Initial learning rate
    lr = 0.01
    min_lr = 1e-8
    max_iter = 10000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(points)
        
        # Update best if improved
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # If gradient is very small, we might have converged
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            break
        
        # Line Search
        step = lr
        improved = False
        # Try up to 10 backtracks
        for _ in range(10):
            # Step in direction of -grad (which is grad of repulsion)
            # Wait, our grad computed above is -sum (xi-xj)/d^2.
            # This is the direction of descent for E?
            # Let's re-verify.
            # E = - sum log d.
            # dE/dxi = - sum (xi-xj)/d^2.
            # Gradient Descent: xi_new = xi - eta * dE/dxi = xi + eta * sum (xi-xj)/d^2.
            # So the update vector is +sum (xi-xj)/d^2.
            # My `grad` variable holds dE/dxi.
            # So I should do points - eta * grad.
            
            new_points = points - step * grad
            
            # Project to sphere
            new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            # If any norm is 0, handle it (shouldn't happen)
            new_norms = np.where(new_norms == 0, 1e-12, new_norms)
            new_points = new_points / new_norms
            
            # Check for collisions (dist < 1e-9)
            # Quick check: min distance
            # This is O(N^2), maybe too slow inside line search?
            # N=100 is small. 10k ops. 10 backtracks = 100k ops. Negligible.
            
            # Compute new energy
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < energy:
                points = new_points
                improved = True
                break
            else:
                step *= 0.5
                
        if not improved:
            # If no improvement found, decay learning rate
            lr *= 0.5
            if lr < min_lr:
                break
        else:
            # If improved, we can keep lr or increase slightly?
            # Keep it stable.
            pass

    # 3. Polish Phase
    # Use very small steps to refine
    # Reset lr to something small
    polish_lr = 1e-5
    polish_max_iter = 5000
    
    for it in range(polish_max_iter):
        if time.time() > deadline:
            break
            
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Simple step without line search for speed, assuming we are close to min
        # Or use a very aggressive line search
        new_points = points - polish_lr * grad
        new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_norms = np.where(new_norms == 0, 1e-12, new_norms)
        new_points = new_points / new_norms
        
        new_energy, _ = _energy_and_grad(new_points)
        
        if new_energy < energy:
            points = new_points
        else:
            polish_lr *= 0.5
            if polish_lr < 1e-12:
                break

    # Final validation
    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Check minimum distance
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Collision detected, fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
