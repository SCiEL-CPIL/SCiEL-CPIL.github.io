import numpy as np
import time
from scipy.optimize import minimize

N = 150

def _energy_and_grad(x_flat):
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d_safe = np.maximum(d, 1e-15)
    E = -np.sum(np.log(d_safe))
    
    dist_sq = dist**2
    dist_sq_safe = np.where(dist_sq > 0, dist_sq, 1.0)
    
    grad_raw = -diff / dist_sq_safe[:, :, np.newaxis]
    grad = np.sum(grad_raw, axis=1)
    
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad_tangent = grad - dot * x
    
    return E, grad_tangent.flatten()

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init():
    x = np.random.normal(size=(N, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_x = None
    best_energy = np.inf
    
    x_fib = _fibonacci_init()
    
    attempts = 0
    while time.time() < deadline:
        # Cycle through initialization strategies
        if attempts % 4 == 0:
            noise = 0.005 * (1 + attempts * 0.05)
            x0 = x_fib + np.random.normal(0, noise, size=(N, 3))
        elif attempts % 4 == 1:
            x0 = _random_init()
        elif attempts % 4 == 2:
            noise = 0.1 * (1 + attempts * 0.05)
            x0 = x_fib + np.random.normal(0, noise, size=(N, 3))
        else:
            # Halton-like or just another random
            x0 = _random_init()
            
        x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        
        try:
            res = minimize(
                _energy_and_grad,
                x0.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_x = res.x.reshape(N, 3)
                best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        except Exception:
            pass
            
        attempts += 1
        
    # Final polish if we have time
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(
                _energy_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_x = res_polish.x.reshape(N, 3)
                best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        except Exception:
            pass

    if best_x is None:
        best_x = x_fib
        
    return best_x
