import numpy as np
import math
import time

N = 150

def _energy_and_grad(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    terms = -diff / dist_sq_safe[:, :, np.newaxis]
    terms = np.where(mask[:, :, np.newaxis], terms, 0.0)
    grad = np.sum(terms, axis=1)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[i_idx, j_idx]
    d_sq_upper = np.maximum(d_sq_upper, 1e-15)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    return energy, grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def _optimize_riemannian_cg(X_init, max_iter=1000, tol=1e-10):
    X = _project_to_sphere(X_init.copy())
    energy, grad = _energy_and_grad(X)
    grad_proj = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    
    d = -grad_proj
    grad_old = grad_proj.copy()
    
    for k in range(max_iter):
        if np.linalg.norm(grad_proj) < tol:
            break
        
        d = d - np.sum(d * X, axis=1, keepdims=True) * X
        
        if np.linalg.norm(d) < 1e-12:
            d = -grad_proj
            
        alpha = 1.0
        for _ in range(20):
            X_new = _project_to_sphere(X + alpha * d)
            energy_new, grad_new = _energy_and_grad(X_new)
            
            if energy_new < energy:
                slope = np.sum(grad_proj * d)
                if energy_new < energy + 1e-4 * alpha * slope:
                    break
            alpha *= 0.5
            
        X = X_new
        energy = energy_new
        grad = grad_new
        grad_proj = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        if k > 0:
            beta = np.sum(grad_proj**2) / (np.sum(grad_old**2) + 1e-15)
            beta = min(beta, 1.0)
            d = -grad_proj + beta * d
        else:
            d = -grad_proj
            
        grad_old = grad_proj.copy()
        
        if k % 10 == 9:
            d = -grad_proj
            grad_old = grad_proj.copy()

    return X, energy

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_X = None
    best_energy = np.inf
    
    X = _fibonacci_sphere(N)
    X, energy = _optimize_riemannian_cg(X, max_iter=2000)
    best_X = X
    best_energy = energy
    
    rng = np.random.default_rng(42)
    perturb_scale = 0.1
    
    while time.time() < deadline:
        noise = rng.normal(0, perturb_scale, (N, 3))
        X_perturbed = _project_to_sphere(best_X + noise)
        
        X_new, energy_new = _optimize_riemannian_cg(X_perturbed, max_iter=1000)
        
        if energy_new < best_energy:
            best_energy = energy_new
            best_X = X_new
            
    return best_X
