"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize logarithmic energy of N points on the unit sphere.
    Strategy: Start from a Fibonacci lattice, then perform gradient descent
    on the sphere using the analytic gradient of the log-energy.
    Use a variable step size (line search) and a time deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Initial Configuration: Fibonacci Spiral
    # This provides a good, well-distributed starting point.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms

    # Precompute indices for pairs
    # We need gradients for each point i: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Energy E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = - sum_{j != i} (1/|x_i - x_j|) * (x_i - x_j) / |x_i - x_j| = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Wait, let's re-verify the sign.
    # E = sum log(1/d) = - sum log(d).
    # dE/dx_i = - sum_{j!=i} (1/d_{ij}) * (x_i - x_j)/d_{ij} = - sum_{j!=i} (x_i - x_j) / d_{ij}^2.
    # We want to minimize E, so we move in the direction of -grad E.
    # Direction = sum_{j!=i} (x_i - x_j) / d_{ij}^2.
    # This is the standard repulsive force.

    def compute_grad(x):
        """
        Compute the gradient of the energy with respect to each point on the sphere.
        Returns the gradient projected onto the tangent space (i.e., grad - <grad, x>x).
        Actually, for constrained optimization on the sphere, we can just use the
        Euclidean gradient and project it to the tangent space for the update,
        or use the formula: update = x + alpha * grad_tangent.
        The Euclidean gradient of E wrt x_i is G_i = - sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2.
        We want to descend, so direction is -G_i = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2.
        Project this onto the tangent space: T_i = D_i - <D_i, x_i>x_i.
        Since <x_i - x_j, x_i> = 1 - x_i·x_j, and <x_i - x_j, x_j> = x_i·x_j - 1.
        Note that D_i is orthogonal to x_i?
        <sum (x_i - x_j)/d^2, x_i> = sum (1 - x_i·x_j)/d^2. This is not necessarily 0.
        So we must project.
        """
        # Distances
        # x shape (N, 3)
        # diff shape (N, N, 3) -> x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # d^2
        d2 = np.sum(diff * diff, axis=2)
        # Avoid division by zero (diagonal)
        np.fill_diagonal(d2, 1.0) # Placeholder, will mask later
        
        # Term (x_i - x_j) / d^2
        # We need sum over j for each i.
        # grad_term[i, j, :] = diff[i, j, :] / d2[i, j]
        # But we only want j != i.
        
        # Efficient calculation:
        # G_i = sum_{j!=i} (x_i - x_j) / d2_{ij}
        # Let's compute the matrix of 1/d2
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # Sum over j: sum_j (x_i - x_j) * inv_d2[i,j]
        # = x_i * sum_j inv_d2[i,j] - sum_j x_j * inv_d2[i,j]
        
        # sum_inv_d2[i] = sum_j inv_d2[i,j]
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        
        # sum_x_weighted[i] = sum_j x_j * inv_d2[i,j]
        # This is inv_d2 @ x
        sum_x_weighted = inv_d2 @ x
        
        # Euclidean Gradient of Energy (negative of force)
        # G_i = - ( x_i * sum_inv_d2[i] - sum_x_weighted[i] )
        # Force F_i = x_i * sum_inv_d2[i] - sum_x_weighted[i]
        # We want to move in direction F_i (repulsion).
        force = x * sum_inv_d2[:, np.newaxis] - sum_x_weighted
        
        # Project to tangent space
        # <force, x_i>
        proj_coeff = np.sum(force * x, axis=1)
        tangent_grad = force - proj_coeff[:, np.newaxis] * x
        
        return tangent_grad

    def compute_energy(x):
        """Compute the total logarithmic energy."""
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d2_masked = np.where(mask, d2, 1.0)
        d = np.sqrt(d2_masked)
        # log(1/d) = -log(d)
        # Sum over i<j
        # Sum over all i!=j is 2 * Sum i<j
        total = np.sum(np.log(d2_masked) / 2.0) # sum log(d) for i!=j
        # We want - sum log(d) for i<j
        # sum_{i!=j} log(d_{ij}) = 2 * sum_{i<j} log(d_{ij})
        # Energy = - sum_{i<j} log(d_{ij}) = - 0.5 * sum_{i!=j} log(d_{ij})
        # Note: log(d) = 0.5 log(d2).
        # So sum_{i!=j} log(d_{ij}) = 0.5 sum_{i!=j} log(d2_{ij})
        # Energy = - 0.5 * (0.5 * sum_{i!=j} log(d2_{ij})) = - 0.25 * sum_{i!=j} log(d2_{ij})
        
        # Let's just compute directly to be safe, using fsum for precision if possible, 
        # but numpy sum is usually fine for intermediate steps. 
        # The evaluator uses math.fsum on the final list.
        
        # For optimization, we just need a consistent value.
        # Let's use the vectorized version.
        sum_log_d2 = np.sum(np.log(d2_masked))
        energy = -0.25 * sum_log_d2
        return energy

    # Optimization Loop
    best_x = x.copy()
    best_energy = compute_energy(x)
    
    # Adaptive step size
    step_size = 0.01
    max_step_size = 0.1
    min_step_size = 1e-5
    patience = 0
    max_patience = 50
    
    current_energy = best_energy
    
    while time.time() < deadline:
        grad = compute_grad(x)
        
        # Normalize gradient to determine a reasonable step?
        # Or just use a fixed heuristic step size adjusted by line search.
        
        # Simple backtracking line search
        alpha = step_size
        improved = False
        for _ in range(10):
            # Update
            x_new = x + alpha * grad
            # Renormalize to sphere
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            
            # Check validity (min distance)
            # If points are too close, reject
            # We can check min distance efficiently? 
            # For now, just compute energy. If it's huge (due to log(small)), it will be rejected.
            
            e_new = compute_energy(x_new)
            
            if e_new < current_energy - 1e-12:
                x = x_new
                current_energy = e_new
                improved = True
                # Increase step size slightly if we improved significantly?
                # step_size = min(max_step_size, step_size * 1.1)
                break
            else:
                alpha *= 0.5
                if alpha < min_step_size:
                    break
        
        if not improved:
            # Decrease step size
            step_size *= 0.5
            patience += 1
            if patience > max_patience:
                break
        else:
            patience = 0
            # Keep step size or increase slightly
            step_size = min(max_step_size, step_size * 1.05)
            
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = x.copy()
            
        # Check for convergence (gradient norm small)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break

    # Final Polish: Use scipy if available and time permits?
    # The task says standard library, NumPy, SciPy are available.
    # Let's try a final L-BFGS-B step if we have time and scipy is imported.
    # But importing scipy inside construct might be slow. 
    # Let's stick to the custom gradient descent for robustness and speed.
    
    # Ensure final normalization
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x
# EVOLVE-BLOCK-END
