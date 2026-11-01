import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its Euclidean gradient.
    X: (N, 3) array of points on the unit sphere.
    Returns:
        energy: float
        grad: (N, 3) array of Euclidean gradients.
    """
    # Pairwise distances
    # Using cdist for potential speedup and stability
    dists = cdist(X, X)
    
    # Mask diagonal to avoid division by zero in log
    # We need sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Let's compute the sum over i != j
    np.fill_diagonal(dists, 1.0) # Placeholder to avoid log(0)
    
    # Clip small distances to avoid -inf
    d_safe = np.maximum(dists, 1e-12)
    
    log_dists = np.log(d_safe)
    # Sum over all i != j
    total_log_sum = np.sum(log_dists)
    energy = -0.5 * total_log_sum
    
    # Gradient
    # E = -0.5 * sum_{i!=j} log(||x_i - x_j||)
    # dE/dx_i = -0.5 * sum_{j!=i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
    #         = -0.5 * sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    
    # We need (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    
    # inv_dists_sq = 1 / d_ij^2 for i != j, 0 for i == j
    inv_dists_sq = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    # grad_i = -0.5 * sum_j diff[i,j] * inv_dists_sq[i,j]
    weighted_diff = diff * inv_dists_sq[:, np.newaxis]
    grad = -0.5 * np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _tangent_projection(X, grad):
    """Project gradient onto the tangent space of the sphere at X."""
    # grad_t = grad - (grad . x) x
    proj = np.sum(grad * X, axis=1, keepdims=True)
    return grad - proj * X

def _slerp(X, V, t):
    """
    Move points X along tangent vectors V by step size t on the sphere.
    V is in the tangent space (V . X = 0).
    This is effectively: X_new = cos(t) * X + sin(t) * (V / ||V||) * ||V|| 
    Wait, if V is a tangent vector, the geodesic is X(t) = cos(t) X(0) + sin(t) T
    where T is the unit tangent direction.
    If we treat V as the velocity vector, the step is:
    angle = t * ||V||
    T = V / ||V||
    X_new = cos(angle) * X + sin(angle) * T
    
    However, for small t, X_new ≈ X + t*V.
    Let's implement the exact geodesic step.
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    T = V / safe_norms
    
    angles = t * norms
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    X_new = cos_a * X + sin_a * T
    
    # Re-normalize to be safe against numerical drift
    X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
    return X_new

def _riemannian_trust_region(X_init, time_limit=100.0):
    """
    Implement a simple Riemannian Trust Region method.
    1. Compute gradient in tangent space.
    2. Solve for step V in tangent space (using Conjugate Gradient on the Hessian approximation or just Newton step if we had Hessian, but we only have grad. 
       Since we don't have Hessian, we can use a simple Gradient Descent with line search, or approximate Newton using BFGS updates on the manifold?
       Actually, standard Riemannian L-BFGS is complex. 
       Let's use a simpler approach: Riemannian Gradient Descent with adaptive step size (Backtracking Line Search).
       This is robust and easy to implement correctly.
    """
    X = X_init.copy()
    E, grad = _energy_and_grad(X)
    grad_t = _tangent_projection(X, grad)
    
    # Initial step size
    step_size = 0.01
    start_time = time.time()
    
    max_iter = 5000
    for k in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-10:
            break
        
        # Line search (Backtracking)
        alpha = step_size
        E_new, _ = _energy_and_grad(X) # Current E
        
        # We want to decrease energy.
        # Armijo condition: E(X + alpha V) <= E(X) + c * alpha * grad_t . V
        # Here V = -grad_t (descent direction)
        # So we want E(X_new) <= E(X) - c * alpha * ||grad_t||^2
        
        c = 1e-4
        rho = 0.5
        max_ls = 20
        
        success = False
        for _ in range(max_ls):
            X_trial = _slerp(X, -grad_t, alpha)
            E_trial, _ = _energy_and_grad(X_trial)
            
            if E_trial <= E + c * alpha * (grad_t * (-grad_t)).sum() : # grad_t dot -grad_t = -||grad_t||^2
                # Wait: E_trial <= E + c * alpha * (grad . step_dir)
                # step_dir = -grad_t
                # grad . step_dir = -||grad_t||^2
                # Condition: E_trial <= E - c * alpha * ||grad_t||^2
                if E_trial <= E - c * alpha * np.sum(grad_t**2):
                    success = True
                    break
            alpha *= rho
        
        if success:
            X = X_trial
            E = E_trial
            # Recompute gradient for next step
            E, grad = _energy_and_grad(X)
            grad_t = _tangent_projection(X, grad)
            
            # Increase step size slightly if we succeeded quickly? Or keep it.
            # Let's keep it stable.
        else:
            # If line search fails, reduce step size significantly
            step_size *= 0.5
            if step_size < 1e-10:
                break

    return X, E

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_X = None
    best_energy = float('inf')
    
    # Strategy: 
    # 1. Try a few random starts
    # 2. Try a Fibonacci start
    # 3. Use Riemannian Gradient Descent with Line Search
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(_random_sphere(N, seed=rng.randint(100000)))
    
    for i, X_start in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Allocate time based on remaining budget
            # If we have 145s total, and we are at start i, we have roughly (145 - elapsed) / remaining_starts
            # But simple fixed limit per run is safer.
            time_limit_for_run = min(20.0, deadline - time.time() - 5.0)
            
            if time_limit_for_run < 1.0:
                continue
                
            X_opt, E_opt = _riemannian_trust_region(X_start, time_limit=time_limit_for_run)
            
            # Verify
            E_check, _ = _energy_and_grad(X_opt)
            
            if E_check < best_energy:
                best_energy = E_check
                best_X = X_opt.copy()
                
        except Exception:
            continue
            
    # If we still have time, do a final polish on the best solution
    if best_X is not None and time.time() < deadline - 10.0:
        try:
            time_limit_polish = min(30.0, deadline - time.time() - 2.0)
            X_polished, E_polished = _riemannian_trust_region(best_X, time_limit=time_limit_polish)
            E_check, _ = _energy_and_grad(X_polished)
            if E_check < best_energy:
                best_energy = E_check
                best_X = X_polished.copy()
        except Exception:
            pass

    if best_X is None:
        best_X = _fibonacci_sphere(N)
        
    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
