import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_only(x):
    """Fast energy computation using pairwise distance matrix."""
    # Use dot product trick: |x_i - x_j|^2 = 2 - 2*x_i.x_j
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1e-30)
    dist_sq = np.maximum(dist_sq, 1e-30)
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    return -math.fsum(np.log(d).tolist())

def _energy_and_grad(x_flat):
    """
    Calculate energy and gradient (projected to tangent space).
    """
    x = x_flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_proj = x / norms
    
    # Pairwise squared distances via dot product
    dot = x_proj @ x_proj.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1e-30)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(dist_sq[iu])
    e = -math.fsum(np.log(d_pairs).tolist())
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x_proj
    
    # Raw gradient
    grad_raw = -x_proj * w_sum[:, np.newaxis] + w_x
    
    # Project to tangent space: g_t = g - (g . x) x
    g_dot_x = np.sum(grad_raw * x_proj, axis=1, keepdims=True)
    grad_tangent = grad_raw - g_dot_x * x_proj
    
    return e, grad_tangent.flatten()

def _grad_tangent(x):
    """Compute tangent-space gradient."""
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1e-30)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    g = -x * w_sum[:, np.newaxis] + w_x
    g_dot_x = np.sum(g * x, axis=1, keepdims=True)
    g_tangent = g - g_dot_x * x
    return g_tangent

def _polish(x, max_iter=500, tol=1e-15, lr_init=0.05, lr_max=0.15):
    """
    Gradient descent with backtracking line search on the sphere.
    Two-phase: larger steps early, smaller steps late.
    """
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    curr_e = _energy_only(x)
    step = lr_init
    for it in range(max_iter):
        g = _grad_tangent(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        alpha = step
        improved = False
        for _ in range(40):
            x_new = x - alpha * g
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            norms_new = np.where(norms_new < 1e-12, 1.0, norms_new)
            x_new = x_new / norms_new
            
            new_e = _energy_only(x_new)
            if new_e < curr_e - 1e-4 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        # Decay step size
        if it % 30 == 29 and step > 0.01:
            step *= 0.75
            
    return x

def _polish_fine(x, max_iter=2000, tol=1e-16):
    """
    Fine polish: very small steps for final convergence.
    """
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    curr_e = _energy_only(x)
    step = 0.01
    for it in range(max_iter):
        g = _grad_tangent(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        alpha = step
        improved = False
        for _ in range(50):
            x_new = x - alpha * g
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            norms_new = np.where(norms_new < 1e-12, 1.0, norms_new)
            x_new = x_new / norms_new
            
            new_e = _energy_only(x_new)
            if new_e < curr_e - 1e-5 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        if it % 50 == 49 and step > 0.002:
            step *= 0.8
            
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin
    
    def time_left():
        return deadline - time.time()

    # Initializations
    rng = np.random.default_rng(42)
    all_inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    all_inits.append(x_fib)
    
    # 2. Fibonacci + small noise (multiple scales)
    for noise_scale in [0.003, 0.01, 0.03, 0.08]:
        noise = rng.normal(0, noise_scale, (N, 3))
        x_fib_noise = x_fib + noise
        x_fib_noise /= np.linalg.norm(x_fib_noise, axis=1, keepdims=True)
        all_inits.append(x_fib_noise)
    
    # 3. Random initializations with different seeds
    num_randoms = 20
    for k in range(num_randoms):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        all_inits.append(x_rand)
    
    # 4. Clustered initializations (some points near each other, some spread)
    for k in range(5):
        centers = rng.normal(size=(15, 3))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        x_clustered = np.empty((N, 3))
        for c_idx, center in enumerate(centers):
            idxs = range(c_idx * 10, c_idx * 10 + 10)
            pts = center + rng.normal(0, 0.1, (10, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            x_clustered[idxs] = pts
        all_inits.append(x_clustered)

    best_x = all_inits[0].copy()
    best_energy = _energy_only(best_x)

    for idx, x0 in enumerate(all_inits):
        if time_left() < 15.0:
            break
        
        try:
            # L-BFGS-B with analytic gradient
            result = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50},
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            # Coarse polish
            if time_left() > 8.0:
                x_opt = _polish(x_opt, max_iter=300, tol=1e-14, lr_init=0.08, lr_max=0.15)
            
            # Fine polish
            if time_left() > 4.0:
                x_opt = _polish_fine(x_opt, max_iter=1000, tol=1e-16)
            
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception:
            continue

    # Final verification: ensure all points are on unit sphere and finite
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    # Re-normalize after any zeroing
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    
    return best_x
