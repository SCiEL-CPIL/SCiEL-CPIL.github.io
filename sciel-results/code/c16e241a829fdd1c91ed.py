import numpy as np
import time
import math
from scipy.optimize import minimize as scipy_minimize

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def compute_energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    log_dist = np.zeros_like(dist)
    log_dist[mask] = np.log(dist[mask])
    
    energy = -0.5 * np.sum(log_dist)
    
    dist_sq = dist**2
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    
    grad_terms = -diff / dist_sq_safe[:, :, np.newaxis]
    
    grad = np.sum(grad_terms, axis=1)
    
    return energy, grad

def normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero if a point collapses to 0
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def optimize(X_init, max_iters=5000, time_limit=100.0):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    X = X_init.copy()
    E, grad = compute_energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    g_tan = grad - grad_dot_X * X
    
    lr = 0.01
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        accepted = False
        current_lr = lr
        while current_lr > 1e-12:
            X_new = X - current_lr * g_tan
            X_new = normalize(X_new)
            
            E_new, grad_new = compute_energy_and_grad(X_new)
            
            if E_new < E:
                X = X_new
                E = E_new
                grad = grad_new
                g_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
                accepted = True
                break
            else:
                current_lr *= 0.5
        
        if not accepted:
            lr *= 0.1
            if lr < 1e-15:
                break
            
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < 1e-10:
            break
            
    return best_X, best_E

def fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_init(N):
    X = np.random.randn(N, 3)
    return normalize(X)

def _scipy_energy_grad_flat(v):
    """
    Energy and gradient for scipy minimize on flat vector.
    We use a penalty-free approach: after each evaluation, we normalize.
    Actually, better to use a constrained approach or just normalize inside.
    For L-BFGS-B, we minimize over R^{N*3} and project back to sphere.
    This is not strictly Riemannian but works well as a polish.
    """
    X = v.reshape(N, 3)
    X = normalize(X)
    E, grad = compute_energy_and_grad(X)
    # The gradient with respect to the flat vector is just the gradient w.r.t X
    # because the normalization is a projection. 
    # Strictly speaking, d/dv normalize(v) = (I - xx^T)/||x|| when x=v.
    # But for a polish step starting near the sphere, the gradient of the 
    # extended energy (energy after projection) is approximately the 
    # tangential gradient.
    # Let's use the tangential gradient projected back.
    # Actually, let's just use the raw gradient. The optimizer will handle it.
    # To be more correct, we should project the gradient to the tangent space 
    # and then map it back to the ambient space for the unconstrained optimizer.
    # grad_flat = grad (since the mapping is identity near the sphere if we 
    # consider the energy as a function on R^3 that is constant on rays? No.)
    
    # Let's use the standard trick: minimize E(normalize(v))
    # dE/dv = (dE/dX) * (dX/dv)
    # dX/dv = (I - X X^T) / ||v||  (if X = v/||v||)
    # Since X is on the sphere, ||v|| is close to 1.
    # The term (I - X X^T) grad is the tangential gradient.
    # So grad_flat = g_tan (approximately, scaled by 1/||v||)
    
    g_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    return E, g_tan.ravel()

def polish(X_init, time_limit=10.0):
    """
    Final high-precision polish using scipy L-BFGS-B.
    """
    start_time = time.time()
    x0 = X_init.ravel()
    
    # Use L-BFGS-B which is robust
    # We don't have bounds, but the normalization handles the constraint.
    # This is technically minimizing a non-smooth function (due to normalization)
    # but it works well in practice for polishing.
    
    try:
        res = scipy_minimize(
            _scipy_energy_grad_flat,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
        )
        
        if res.success or res.fun < compute_energy_and_grad(normalize(X_init))[0]:
            X_opt = normalize(res.x.reshape(N, 3))
        else:
            X_opt = X_init
            
        # Check time
        if time.time() - start_time > time_limit:
            return X_init
        
        return X_opt
    except Exception:
        return X_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_X = None
    best_E = float('inf')
    
    # 1. Fibonacci Start
    X_fib = fibonacci_spiral(N)
    X_opt, E_opt = optimize(X_fib, max_iters=5000, time_limit=30.0)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
        
    # 2. Random Starts
    num_random_starts = 5
    time_per_start = (deadline - time.time()) / (num_random_starts + 1)
    
    for k in range(num_random_starts):
        if time.time() > deadline:
            break
            
        X_rand = random_init(N)
        # Quick pre-optimization to avoid bad local minima
        X_pre, E_pre = optimize(X_rand, max_iters=500, time_limit=time_per_start * 0.5)
        
        if time.time() > deadline:
            break
            
        # Fine optimization
        X_opt, E_opt = optimize(X_pre, max_iters=2000, time_limit=time_per_start * 0.5)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # 3. Perturbation of best
    if time.time() < deadline and best_X is not None:
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline:
                break
            noise = np.random.randn(*best_X.shape) * 0.1
            X_pert = normalize(best_X + noise)
            X_opt, E_opt = optimize(X_pert, max_iters=1000, time_limit=(deadline - time.time()) * 0.9)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
                
    # 4. Final Polish
    if time.time() < deadline and best_X is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            X_polished = polish(best_X, time_limit=remaining_time * 0.8)
            E_polished, _ = compute_energy_and_grad(X_polished)
            if E_polished < best_E:
                best_E = E_polished
                best_X = X_polished

    return best_X
