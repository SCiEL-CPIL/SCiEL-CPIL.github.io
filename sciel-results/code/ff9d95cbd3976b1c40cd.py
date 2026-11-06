import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate energy and gradient (tangent-space gradient for sphere constraint).
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
    
    # Raw gradient of energy with respect to x
    grad_raw = -x_proj * w_sum[:, np.newaxis] + w_x
    
    # Project to tangent space: grad_tangent = grad_raw - (grad_raw . x) x
    grad_dot_x = np.sum(grad_raw * x_proj, axis=1, keepdims=True)
    grad_tangent = grad_raw - grad_dot_x * x_proj
    
    return e, grad_tangent.flatten()

def _polish(x, max_iter=300, tol=1e-15):
    """
    Gradient descent with backtracking line search on the sphere.
    Two-stage: first with larger steps, then fine-tuning.
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
    
    # Stage 1: Coarse optimization
    step = 0.1
    for _ in range(max_iter // 2):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = step
        for _ in range(40):
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
    
    # Stage 2: Fine-tuning with smaller steps
    step = 0.01
    for _ in range(max_iter // 2):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = step
        for _ in range(40):
            # Move in direction -g
            x_new = x - alpha * g
            # Retract to sphere
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            
            new_e = energy(x_new)
            if new_e < curr_e - 1e-6 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                break
            alpha *= 0.5
        else:
            break
            
    return x

def _fibonacci_spiral(n, offset=0.0):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s buffer for safety
    
    def check_time():
        if time.time() > deadline:
            raise TimeoutError

    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral (standard)
    x_fib = _fibonacci_spiral(N)
    inits.append(x_fib)
    
    # 2. Fibonacci Spiral with offset
    x_fib_off = _fibonacci_spiral(N, offset=0.25)
    inits.append(x_fib_off)
    
    # 3. Fibonacci + small noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, (N, 3))
    x_fib_noise = x_fib + noise
    x_fib_noise /= np.linalg.norm(x_fib_noise, axis=1, keepdims=True)
    inits.append(x_fib_noise)
    
    # 4. Fibonacci + medium noise
    noise2 = rng.normal(0, 0.05, (N, 3))
    x_fib_noise2 = x_fib + noise2
    x_fib_noise2 /= np.linalg.norm(x_fib_noise2, axis=1, keepdims=True)
    inits.append(x_fib_noise2)
    
    # 5. Random initializations
    num_randoms = 15
    for k in range(num_randoms):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
    
    best_x = inits[0]
    best_energy = np.inf

    for x0 in inits:
        check_time()
        
        # If we have little time left, skip
        if time.time() > deadline - 8.0:
            break

        try:
            # Use L-BFGS-B with proper tangent-space gradient
            result = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 40},
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            # Polish with two-stage approach
            if time.time() < deadline - 5.0:
                x_opt = _polish(x_opt, max_iter=200, tol=1e-15)
            
            e_opt = _energy_and_grad(x_opt.flatten())[0]
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception:
            continue

    return best_x
