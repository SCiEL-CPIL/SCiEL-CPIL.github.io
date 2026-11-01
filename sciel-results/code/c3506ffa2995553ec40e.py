import numpy as np
import time
from scipy.optimize import minimize

N = 150

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def energy_and_grad(v):
    X = v.reshape(-1, 3)
    # Compute pairwise distances
    # D[i, j] = ||X[i] - X[j]||
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) and division by zero
    d_safe = np.maximum(d, 1e-12)
    
    # Energy: -sum_{i<j} log(d_ij)
    # We only sum i < j
    iu = np.triu_indices(N, k=1)
    E = -np.sum(np.log(d_safe[iu]))
    
    # Gradient
    # g_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    # We can compute this using the diff matrix
    # term_ij = (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = sum_k (x_ik - x_jk)^2
    d2 = d_safe ** 2
    # diff is (N, N, 3)
    # We need to divide diff by d2[:, :, np.newaxis]
    # But set diagonal to 0
    inv_d2 = np.zeros_like(d2)
    mask = d2 > 1e-24
    inv_d2[mask] = 1.0 / d2[mask]
    
    # term[i, j] = diff[i, j] * inv_d2[i, j]
    terms = diff * inv_d2[:, :, np.newaxis]
    
    # g_i = - sum_j terms[i, j]
    G = -np.sum(terms, axis=1)
    
    # Penalty gradient
    # P = mu * sum (||x_i||^2 - 1)^2
    # dP/dx_i = 4 * mu * (||x_i||^2 - 1) * x_i
    mu = 100.0
    norms_sq = np.sum(X**2, axis=1)
    G += 4.0 * mu * (norms_sq - 1.0)[:, np.newaxis] * X
    
    return E, G.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = np.inf
    best_X = None
    
    # Initial guess: Fibonacci spiral
    X0 = fibonacci_sphere(N)
    
    # We will do multiple restarts with jitter
    # Jitter amount should be small
    rng = np.random.default_rng(42)
    
    n_restarts = 50 # Max restarts, but we will check time
    
    for k in range(n_restarts):
        # Jitter
        if k == 0:
            X_init = X0.copy()
        else:
            # Small random perturbation
            # Project back to sphere
            X_init = X0 + 0.1 * rng.standard_normal((N, 3))
            norms = np.linalg.norm(X_init, axis=1, keepdims=True)
            X_init = X_init / norms
        
        v_init = X_init.flatten()
        
        # Define objective for scipy
        def obj_and_grad(v):
            E, G = energy_and_grad(v)
            return E, G
        
        try:
            res = minimize(
                obj_and_grad,
                v_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            X_opt = res.x.reshape(-1, 3)
            E_opt = res.fun
            
            # Check if better
            if E_opt < best_energy:
                best_energy = E_opt
                best_X = X_opt
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break
            
    if best_X is None:
        return X0
    return best_X
