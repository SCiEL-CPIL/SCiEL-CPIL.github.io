import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy_and_grad(pts):
        # pts: (N, 3)
        # Compute pairwise squared distances
        # D_ij = ||x_i - x_j||^2
        # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero on diagonal
        # Diagonal is 0. We mask it out.
        mask = ~np.eye(N, dtype=bool)
        
        # Gradient: sum_{j != i} (x_i - x_j) / sq_dists[i, j]
        # diff is (x_i - x_j)
        # We need to divide each vector diff[i, j] by sq_dists[i, j]
        # sq_dists has shape (N, N). We need to broadcast it to (N, N, 1)
        inv_sq_dists = np.zeros_like(sq_dists)
        inv_sq_dists[mask] = 1.0 / sq_dists[mask]
        
        # grad[i] = sum_j diff[i, j] * inv_sq_dists[i, j]
        # diff is (N, N, 3), inv_sq_dists is (N, N) -> broadcast to (N, N, 1)
        grad = np.sum(diff * inv_sq_dists[:, :, np.newaxis], axis=1)
        
        # Energy: sum_{i < j} -log(sqrt(sq_dists[i, j])) = -0.5 * sum_{i < j} log(sq_dists[i, j])
        # Or sum_{i != j} -0.5 log(sq_dists[i, j]) / 2 ? No.
        # E = sum_{i < j} log(1 / d_ij) = sum_{i < j} - log(d_ij)
        # d_ij = sqrt(sq_dists[i, j])
        # log(d_ij) = 0.5 log(sq_dists[i, j])
        # E = sum_{i < j} -0.5 log(sq_dists[i, j])
        
        # We sum over all i != j, which is 2 * sum_{i < j}
        # So E = -0.25 * sum_{i != j} log(sq_dists[i, j])
        
        # Calculate log of squared distances for off-diagonal elements
        # sq_dists[mask] contains squared distances for i != j
        log_sq_dists = np.log(sq_dists[mask])
        energy = -0.25 * np.sum(log_sq_dists)
        
        return energy, grad

    # Initial energy
    current_energy, grad = energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    alpha = 1.0 # Initial step size
    min_alpha = 1e-10
    
    # Optimization loop
    while time.time() < deadline:
        # Project gradient onto tangent space
        # proj_v = v - (v . x) x
        dot = np.sum(pts * grad, axis=1)
        tangent_grad = grad - dot[:, np.newaxis] * pts
        
        # Check for convergence (optional, but good)
        # if np.max(np.linalg.norm(tangent_grad, axis=1)) < 1e-8:
        #     break
        
        # Backtracking line search
        found_step = False
        while alpha > min_alpha:
            # Propose new points
            new_pts = pts - alpha * tangent_grad
            # Renormalize
            norms = np.linalg.norm(new_pts, axis=1)
            # Handle case where norm is 0 (should not happen)
            new_pts = new_pts / norms[:, np.newaxis]
            
            new_energy, new_grad = energy_and_grad(new_pts)
            
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                grad = new_grad
                found_step = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
                # Increase alpha slightly if we took a full step? Or keep it.
                # Usually keep or increase slowly. Let's keep it to be safe, or increase by 1.1
                # alpha = min(alpha * 1.1, 1.0) 
                break
            else:
                alpha *= 0.5
        
        if not found_step:
            # If we couldn't find a step even with tiny alpha, we might be stuck or at a minimum
            # Break if alpha is too small
            if alpha < min_alpha:
                break
            # Otherwise, maybe reset alpha? Or break.
            # If we can't decrease energy, we are likely at a local min.
            break

    return best_pts
