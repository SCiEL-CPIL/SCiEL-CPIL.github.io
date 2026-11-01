import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its Euclidean gradient.
    X: (N, 3) array of points on the unit sphere.
    
    Energy E = sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    
    Gradient w.r.t x_k:
    g_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    
    Returns:
        energy: float
        grad: (N, 3) array of Euclidean gradients.
    """
    # Pairwise differences: diff[i, j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    
    # Distances squared
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i < j of -log(dist_ij)
    # We can compute sum over all i != j and divide by 2
    # log(1/d) = -log(d)
    # Sum_{i!=j} -log(d_ij) / 2
    # Note: log(d_ij) = 0.5 * log(d_ij^2)
    # So -log(d_ij) = -0.5 * log(d_ij^2)
    
    # Create mask for upper triangle to sum exactly i < j
    # Or just sum all and divide by 2.
    # Using all pairs is faster in vectorized numpy.
    mask = ~np.eye(N, dtype=bool)
    
    # log_dist = 0.5 * log(dist_sq)
    log_dist_sq = np.log(dist_sq)
    # We want -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    # Sum_{i<j} log(d_ij^2) = 0.5 * Sum_{i!=j} log(d_ij^2)
    # So Energy = -0.5 * 0.5 * Sum_{i!=j} log(d_ij^2) = -0.25 * Sum_{i!=j} log(d_ij^2)
    
    energy = -0.25 * np.sum(log_dist_sq[mask])
    
    # Gradient
    # g_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    # diff[k, j] = x_k - x_j
    # dist_sq[k, j] = ||x_k - x_j||^2
    # We want sum_j diff[k, j] / dist_sq[k, j]
    
    # Normalize diff by dist_sq
    # inv_dist_sq = 1 / dist_sq
    # Set diagonal to 0 to avoid contribution from j=k
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad[k] = sum_j diff[k, j] * inv_dist_sq[k, j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    v = np.random.randn(n, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _optimize(X0, deadline):
    """
    Run L-BFGS-B optimization from X0 until deadline or convergence.
    """
    current_time = time.time()
    if current_time > deadline:
        return X0, _energy_and_grad(X0)[0]
    
    best_X = X0.copy()
    best_E, _ = _energy_and_grad(X0)
    
    # L-BFGS-B on the Euclidean space, but we must ensure points stay on the sphere.
    # Standard trick: optimize on R^3 and project back to sphere after each step?
    # No, L-BFGS-B assumes a linear space. 
    # Better approach: Use a parameterization that guarantees unit norm, or 
    # use the fact that the gradient of the energy on the sphere is the projection 
    # of the Euclidean gradient onto the tangent space.
    # g_tangent = g_euclidean - (g_euclidean . x) * x
    
    # However, scipy.optimize.minimize does not natively handle manifolds.
    # We can define a custom function that takes parameters, reconstructs points, 
    # and returns energy and gradient.
    # But for N=100 points in R^3, we have 300 variables.
    # If we optimize in R^3 and project, the gradient calculation for the optimizer 
    # must be consistent with the projection.
    
    # A common and robust approach for this specific problem is to use the 
    # "Riemannian L-BFGS" logic manually or use a library. Since we only have 
    # standard SciPy, we will implement a simple Riemannian gradient descent 
    # or use L-BFGS-B with a projection step and corrected gradient.
    
    # Actually, a simpler and very effective method for this specific energy is 
    # to use the Euclidean gradient but project the update back to the sphere.
    # The gradient of the energy restricted to the sphere is the tangential component.
    
    # Let's implement a custom Riemannian L-BFGS-B or just a robust Riemannian 
    # gradient descent with adaptive step size, as L-BFGS-B in Euclidean space 
    # might struggle with the manifold constraint if not handled carefully.
    
    # Alternative: Use the fact that for points on the sphere, we can optimize 
    # using the Euclidean gradient but project the position.
    # Let's try a simple Riemannian Steepest Descent with Barzilai-Borwein step size 
    # or just a fixed small step size with line search.
    
    # Given the time constraint, a robust Riemannian Gradient Descent is safer 
    # than implementing a full Riemannian L-BFGS from scratch.
    
    X = X0.copy()
    E, g_euclid = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    # Project gradient to tangent space
    g_tang = g_euclid - np.sum(g_euclid * X, axis=1, keepdims=True) * X
    
    # Adaptive step size
    step = 0.01
    max_iter = 1000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Update
        X_new = X - step * g_tang
        
        # Project back to sphere
        X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
        
        E_new, g_euclid_new = _energy_and_grad(X_new)
        
        if E_new < E:
            # Accept
            # Simple line search / step size adjustment
            # If we improved significantly, maybe increase step?
            # If we improved slightly, keep step.
            X = X_new
            E = E_new
            g_tang = g_euclid_new - np.sum(g_euclid_new * X, axis=1, keepdims=True) * X
            
            if E < best_E:
                best_E = E
                best_X = X.copy()
                
            # Heuristic step size adjustment
            # If improvement is small, reduce step to polish
            if step > 1e-5:
                 step *= 0.9
        else:
            # Reject
            step *= 0.5
            
            if step < 1e-8:
                break

    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Generate multiple diverse starting points
    # 1. Fibonacci sphere
    # 2. Fibonacci sphere with slight random rotation
    # 3. Random points (less likely to be good, but diverse)
    
    starts = []
    
    # Start 1: Pure Fibonacci
    X_fib = _fibonacci_sphere(N)
    starts.append(X_fib)
    
    # Start 2: Fibonacci with small random perturbations
    X_pert = X_fib.copy()
    noise = np.random.randn(N, 3) * 0.01
    X_pert += noise
    X_pert /= np.linalg.norm(X_pert, axis=1, keepdims=True)
    starts.append(X_pert)
    
    # Start 3: Fibonacci rotated by random angle
    # Generate random rotation matrix
    u, s, vh = np.linalg.svd(np.random.randn(3, 3))
    R = u @ vh
    X_rot = X_fib @ R.T
    starts.append(X_rot)
    
    # Start 4: Another random rotation
    u, s, vh = np.linalg.svd(np.random.randn(3, 3))
    R = u @ vh
    X_rot2 = X_fib @ R.T
    starts.append(X_rot2)
    
    # Start 5: Random points
    X_rand = _random_sphere(N)
    starts.append(X_rand)
    
    best_X_global = None
    best_E_global = float('inf')
    
    # Allocate time to each start
    # We have ~145 seconds.
    # Let's give the first 4 starts more time as they are structured.
    # 1. Fib: 40s
    # 2. Pert: 30s
    # 3. Rot: 30s
    # 4. Rot2: 20s
    # 5. Rand: 10s
    # Plus overhead.
    
    time_alloc = [40.0, 30.0, 30.0, 20.0, 10.0]
    
    for i, X0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        t_end_for_this = time.time() + time_alloc[i]
        # Ensure we don't exceed global deadline
        if t_end_for_this > deadline - 5.0:
            t_end_for_this = deadline - 5.0
            
        X_opt, E_opt = _optimize(X0, t_end_for_this)
        
        if E_opt < best_E_global:
            best_E_global = E_opt
            best_X_global = X_opt.copy()
            
    # Final polish: Run a few more iterations on the best result with very small steps
    if best_X_global is not None and time.time() < deadline - 2.0:
        # Re-run optimization with tighter time limit and smaller initial step
        # Modify _optimize to accept initial step? Or just call again.
        # Let's just call _optimize again on the best result with remaining time
        X_final, E_final = _optimize(best_X_global, deadline - 1.0)
        if E_final < best_E_global:
            best_X_global = X_final
            best_E_global = E_final
            
    # Final normalization
    best_X_global = best_X_global / np.linalg.norm(best_X_global, axis=1, keepdims=True)
    
    return best_X_global
