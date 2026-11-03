"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient(points):
    """
    Compute the gradient of the logarithmic energy on the sphere.
    Returns the tangent vector (negative gradient) to minimize energy.
    """
    n = len(points)
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0) # Avoid division by zero on diagonal
    
    # factor[i,j] = 1 / d2[i,j]
    factor = 1.0 / d2
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j (points[i] - points[j]) / d2[i,j]
    # This is the repulsive force, i.e., -grad(energy)
    force = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot_prod = np.sum(force * points, axis=1)
    tangent_force = force - dot_prod[:, np.newaxis] * points
    
    return tangent_force

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for evaluation overhead
    
    best_points = None
    best_energy = float('inf')
    
    # Initial guess: Fibonacci spiral
    init_points = _fibonacci_sphere(N)
    
    num_restarts = 20 # More restarts with faster convergence
    
    np.random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            current = init_points.copy()
        else:
            # Random perturbation
            noise_std = 0.5 * (1.0 - restart / num_restarts) + 0.1
            current = init_points + np.random.normal(0, noise_std, size=(N, 3))
            current = _project_to_sphere(current)
            
        current_energy = _log_energy(current)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current.copy()
            
        # Gradient Descent with Backtracking Line Search
        max_iters = 5000
        alpha = 0.1
        c1 = 1e-4 # Armijo condition
        rho = 0.5 # Shrink factor
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            grad = _gradient(current)
            
            # If gradient is small, stop
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
                
            # Backtracking line search
            step = alpha * grad
            new_points = _project_to_sphere(current + step)
            new_energy = _log_energy(new_points)
            
            # Armijo condition: new_energy <= current_energy - c1 * alpha * ||grad||^2
            # Note: grad is tangent, so ||grad||^2 is the directional derivative squared roughly
            # Actually, directional derivative is grad . grad = ||grad||^2
            expected_decrease = c1 * alpha * grad_norm * grad_norm
            
            while new_energy > current_energy - expected_decrease and alpha > 1e-8:
                alpha *= rho
                step = alpha * grad
                new_points = _project_to_sphere(current + step)
                new_energy = _log_energy(new_points)
                expected_decrease = c1 * alpha * grad_norm * grad_norm
                
            if alpha <= 1e-8:
                # Stuck, try a random kick or break
                break
                
            current = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current.copy()
                
            # Reset alpha to a larger value if we took a full step, or keep it
            # alpha = min(alpha * 1.5, 0.1) # Optional: increase alpha if step was small?
            
    return best_points
# EVOLVE-BLOCK-END
