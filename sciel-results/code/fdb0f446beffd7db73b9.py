import numpy as np
import time
import math

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
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def optimize(X_init, max_iters=5000, time_limit=100.0, tol=1e-10):
    """
    Riemannian Gradient Descent with adaptive learning rate.
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
        for _ in range(30):
            if current_lr < 1e-14:
                break
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
            lr *= 0.5
            if lr < 1e-15:
                break
        else:
            # Slowly increase learning rate if steps are being accepted
            lr = min(lr * 1.02, 0.1)
            
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
            
    return best_X, best_E

def fibonacci_spiral(N, seed=0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed * 1.0 / N)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_init(N, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 3))
    return normalize(X)

def polish(X, time_limit=10.0, tol=1e-12):
    """
    High-precision polish pass with very tight tolerance.
    """
    X = X.copy()
    E, grad = compute_energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    g_tan = grad - grad_dot_X * X
    
    lr = 0.001
    start_time = time.time()
    
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        accepted = False
        current_lr = lr
        for _ in range(40):
            if current_lr < 1e-16:
                break
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
            lr *= 0.5
            if lr < 1e-18:
                break
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
            
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_X = None
    best_E = float('inf')
    
    # 1. Fibonacci Start with good initial optimization
    X_fib = fibonacci_spiral(N)
    remaining = deadline - time.time()
    if remaining > 5.0:
        X_opt, E_opt = optimize(X_fib, max_iters=3000, time_limit=min(25.0, remaining * 0.5))
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # 2. Multiple diverse random starts
    num_starts = 4
    remaining = deadline - time.time()
    if remaining > 10.0:
        time_per_start = remaining * 0.6 / num_starts
        for k in range(num_starts):
            if time.time() > deadline:
                break
            X_rand = random_init(N, seed=k * 12345 + 67890)
            X_opt, E_opt = optimize(X_rand, max_iters=2000, time_limit=time_per_start)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
    
    # 3. Perturbations of best solution
    remaining = deadline - time.time()
    if remaining > 5.0 and best_X is not None:
        num_perturb = 3
        time_per_pert = remaining * 0.4 / num_perturb
        for k in range(num_perturb):
            if time.time() > deadline:
                break
            noise = np.random.randn(*best_X.shape) * 0.05
            X_pert = normalize(best_X + noise)
            X_opt, E_opt = optimize(X_pert, max_iters=1500, time_limit=time_per_pert)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
    
    # 4. Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 3.0 and best_X is not None:
        polish_time = min(remaining * 0.8, 15.0)
        best_X, best_E = polish(best_X, time_limit=polish_time, tol=1e-12)
    
    return best_X
