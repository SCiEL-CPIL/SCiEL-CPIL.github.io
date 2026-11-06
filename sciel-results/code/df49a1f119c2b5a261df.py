import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate energy and gradient.
    """
    x = x_flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_proj = x / norms
    
    # Pairwise distances
    diff = x_proj[:, np.newaxis, :] - x_proj[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + 1e-30
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = dist[mask]
    
    # Energy
    e = -np.sum(np.log(d_pairs))
    
    # Gradient Calculation (Analytic on the sphere)
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x_proj
    
    grad_sphere = -x_proj * w_sum[:, np.newaxis] + w_x
    
    return e, grad_sphere.flatten()

def _polish(x, max_iter=200, tol=1e-14):
    """
    Gradient descent with backtracking line search on the sphere.
    """
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    def energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + 1e-30
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_pairs = np.sqrt(dist_sq[mask])
        return -np.sum(np.log(d_pairs))
    
    def grad(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + 1e-30
        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)
        w_sum = np.sum(weights, axis=1)
        w_x = weights @ x
        g = -x * w_sum[:, np.newaxis] + w_x
        # Project gradient to tangent space
        g_dot_x = np.sum(g * x, axis=1, keepdims=True)
        g_tangent = g - g_dot_x * x
        return g_tangent

    curr_e = energy(x)
    step = 0.1
    for _ in range(max_iter):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = step
        for _ in range(30):
            # Move in direction -g
            x_new = x - alpha * g
            # Retract to sphere
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            
            new_e = energy(x_new)
            if new_e < curr_e - 1e-4 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                break
            alpha *= 0.5
        else:
            break
            
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    def check_time():
        if time.time() > deadline:
            raise TimeoutError

    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Fibonacci + Noise
    noise = np.random.normal(0, 0.01, (N, 3))
    x_fib_noise = x_fib + noise
    x_fib_noise /= np.linalg.norm(x_fib_noise, axis=1, keepdims=True)
    inits.append(x_fib_noise)
    
    # 3. Random
    rng = np.random.default_rng(42)
    # We will add random inits dynamically in the loop to use time efficiently
    num_randoms = 20
    
    best_x = inits[0]
    best_energy = np.inf

    all_inits = inits[:]
    for k in range(num_randoms):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        all_inits.append(x_rand)

    for x0 in all_inits:
        check_time()
        
        # If we have little time left, skip
        if time.time() > deadline - 5.0:
            break

        try:
            # Use analytic gradient for speed and accuracy
            result = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50},
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            # Polish
            if time.time() < deadline - 2.0:
                x_opt = _polish(x_opt, max_iter=100, tol=1e-14)
            
            e_opt = _energy_and_grad(x_opt.flatten())[0]
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception:
            continue

    return best_x
