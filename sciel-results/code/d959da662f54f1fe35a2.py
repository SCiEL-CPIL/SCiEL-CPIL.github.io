"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    # To avoid log(0), we assume points are distinct. 
    # For numerical stability in gradient, we use dist_sq.
    # log(1/d) = -0.5 * log(d^2)
    
    safe_dist_sq = np.where(mask, dist_sq, 1.0)
    log_terms = -0.5 * np.log(safe_dist_sq)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: The gradient of -log(|x_i - x_j|) wrt x_i is -(x_i - x_j)/|x_i - x_j|^2
    # Since we sum over pairs i<j, the force on i from j is -(x_i - x_j)/d^2 and on j is -(x_j - x_i)/d^2 = (x_i - x_j)/d^2.
    # Wait, let's re-verify.
    # E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = - sum_{j != i} (1/|x_i - x_j|) * (x_i - x_j)/|x_i - x_j| = - sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # This is correct.
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / (dist_sq[mask] + 1e-15) # small epsilon for stability
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _project_to_sphere(x):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms

def _fibonacci_sphere(N, seed=0):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _halton_sphere(N, seed=0):
    """Generate points on the sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    # Use first 3 primes
    primes = [2, 3, 5]
    u = np.array([halton(i+1, p) for i, p in enumerate(primes[:1])]) # Wait, need 3 coords
    u = np.zeros((N, 3))
    for i in range(N):
        for j, p in enumerate(primes):
            u[i, j] = halton(i+1, p)
    
    # Map to sphere using cosine method or similar
    # A simple mapping: 
    # theta = 2*pi*u[:,0]
    # z = 2*u[:,1] - 1
    # r = sqrt(1-z^2)
    # x = r*cos(theta), y=r*sin(theta)
    # This clusters at poles. Better to use inverse CDF for uniform distribution on sphere.
    # z ~ U(-1, 1)
    # theta ~ U(0, 2pi)
    z = 2.0 * u[:, 1] - 1.0
    theta = 2.0 * math.pi * u[:, 0]
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return x

def _random_sphere(N, seed=0):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _riemannian_lbgfs(x0, max_iter=1000, time_limit=10.0):
    """
    Perform Riemannian L-BFGS on the sphere.
    We parameterize the movement in the tangent space.
    """
    x = x0.copy()
    e, g = _log_energy_and_grad_sphere(x)
    
    # Project gradient to tangent space
    g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
    
    # If gradient is small, we are at a critical point
    if np.linalg.norm(g_tan) < 1e-8:
        return x, e
    
    # Use a simple gradient descent step in tangent space with adaptive step size
    # This is a simplified version of Riemannian L-BFGS to keep it robust and fast
    # We use a line search in the exponential map
    
    step_size = 0.01
    min_step = 1e-6
    max_step = 1.0
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Exponential map: x_new = cos(theta)*x + sin(theta)*(g_tan/||g_tan||)
        # Here we move in direction -g_tan
        v = -g_tan
        norm_v = np.linalg.norm(v, axis=1, keepdims=True)
        
        # Avoid zero norm
        valid = norm_v > 1e-10
        v[~valid] = 0
        norm_v[~valid] = 1
        
        v_unit = v / norm_v
        
        # Step size per point? Global step size is better for stability.
        # Let's try a global step size with backtracking line search.
        
        alpha = step_size
        found = False
        for _ in range(20):
            # Move all points by alpha in their respective tangent directions
            # x_new = cos(alpha)*x + sin(alpha)*v_unit
            c = np.cos(alpha)
            s = np.sin(alpha)
            x_new = c * x + s * v_unit
            
            # Ensure projection (numerical error)
            x_new = _project_to_sphere(x_new)
            
            e_new, _ = _log_energy_and_grad_sphere(x_new)
            
            if e_new < e:
                found = True
                break
            else:
                alpha *= 0.5
                if alpha < min_step:
                    break
        
        if found:
            x = x_new
            e = e_new
            step_size = min(step_size * 1.5, max_step)
        else:
            step_size = max(step_size * 0.5, min_step)
            if step_size <= min_step * 1.01:
                break
        
        # Update gradient
        _, g = _log_energy_and_grad_sphere(x)
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        
        if np.linalg.norm(g_tan) < 1e-8:
            break
            
    return x, e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_x = None
    best_energy = float('inf')
    
    # Multiple initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(_fibonacci_sphere(N, seed=0))
    inits.append(_fibonacci_sphere(N, seed=1))
    
    # 2. Halton
    inits.append(_halton_sphere(N, seed=0))
    inits.append(_halton_sphere(N, seed=1))
    
    # 3. Random
    inits.append(_random_sphere(N, seed=0))
    inits.append(_random_sphere(N, seed=1))
    inits.append(_random_sphere(N, seed=2))
    
    # 4. Slightly perturbed Fibonacci
    rng = np.random.default_rng(123)
    x_fib = _fibonacci_sphere(N, seed=0)
    x_pert = x_fib + rng.normal(scale=0.1, size=(N, 3))
    inits.append(_project_to_sphere(x_pert))
    
    # Optimize each initialization
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Determine time budget for this run
        # We have ~100s total. 8 inits. ~12s each?
        # Let's give 10s to each, but stop early if we run out of time.
        time_budget = min(10.0, deadline - time.time() - 2.0)
        if time_budget <= 0:
            break
            
        try:
            x_res, e_res = _riemannian_lbgfs(x0, max_iter=200, time_limit=time_budget)
            
            # Check validity
            if np.all(np.isfinite(x_res)) and np.all(np.linalg.norm(x_res, axis=1) > 1e-9):
                # Check for collisions
                diff = x_res[:, np.newaxis, :] - x_res[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                mask = ~np.eye(N, dtype=bool)
                min_dist = np.min(dist[mask])
                if min_dist > 1e-9:
                    if e_res < best_energy:
                        best_energy = e_res
                        best_x = x_res.copy()
        except Exception:
            pass

    # If we have time left, polish the best result with a more careful optimization
    if best_x is not None and time.time() < deadline - 5:
        # Use scipy minimize on the flattened sphere coordinates with projection?
        # Or just run our Riemannian CG for a bit longer with smaller steps.
        x_polish, e_polish = _riemannian_lbgfs(best_x, max_iter=500, time_limit=deadline - time.time() - 2.0)
        
        if e_polish < best_energy:
            best_x = x_polish
            best_energy = e_polish
            
    # Final normalization and validation
    if best_x is None:
        # Fallback: return Fibonacci
        best_x = _fibonacci_sphere(N, seed=0)
        
    best_x = _project_to_sphere(best_x)
    
    # Final check for invalid points
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    min_dist = np.min(dist[mask])
    
    if min_dist < 1e-9:
        # If collision, try to separate them slightly
        # This is a heuristic repair
        idx_i, idx_j = np.unravel_index(np.argmin(dist[mask]), (N, N))
        # Move idx_j slightly away from idx_i
        v = best_x[idx_i] - best_x[idx_j]
        v /= np.linalg.norm(v)
        best_x[idx_j] = _project_to_sphere(best_x[idx_j] + 1e-8 * v)
        best_x = _project_to_sphere(best_x)
        
    return best_x

# EVOLVE-BLOCK-END
