import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 150
TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _log_energy(X):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    X: (N, 3) array of unit vectors.
    """
    # Pairwise squared distances
    d2 = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)[np.newaxis, :] - 2 * (X @ X.T)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-15))
    
    # Sum over i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    E = -np.sum(np.log(d[mask]))
    return E

def _energy_and_grad(X):
    """
    Compute logarithmic energy and Riemannian gradient.
    X: (N, 3) array of unit vectors.
    Energy: sum_{i<j} log(1/d_ij)
    Grad: Tangent space gradient.
    """
    # Pairwise squared distances
    d2 = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)[np.newaxis, :] - 2 * (X @ X.T)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-15))
    
    # Energy: sum_{i<j} -log(d_ij)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    E = -np.sum(np.log(d[mask]))
    
    # Gradient
    # grad_E_i = sum_{j != i} (X_i - X_j) / d_ij^2
    W = 1.0 / d2
    np.fill_diagonal(W, 0.0)
    
    row_sums = np.sum(W, axis=1)
    WX = W @ X
    
    grad_eucl = row_sums[:, np.newaxis] * X - WX
    
    # Project to tangent space
    proj = np.sum(grad_eucl * X, axis=1)
    grad_r = grad_eucl - proj[:, np.newaxis] * X
    
    return E, grad_r

def _retract(X, P, alpha):
    """
    Retraction: move from X along tangent vector P by alpha, then normalize.
    """
    X_new = X + alpha * P
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X_new / norms

def _exponential_map(X, P):
    """
    Exponential map on the sphere.
    For small steps, this is very similar to retraction.
    """
    # For the sphere, the exponential map is:
    # exp_x(v) = x cos(||v||) + (v/||v||) sin(||v||)
    # But for our purposes, retraction is sufficient.
    return _retract(X, P, 1.0)

def _fibonacci_sphere():
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _halton_sphere():
    """Generate points on sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(max(0, 1 - z*z))
        points.append([r * math.cos(theta), r * math.sin(theta), z])
    return np.array(points)

def _random_sphere(seed):
    """Generate random points on sphere."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _riemannian_cg(X_init, time_budget, max_iter=5000):
    """
    Riemannian Conjugate Gradient method.
    """
    X = X_init.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-15)
    
    E_curr, G_curr = _energy_and_grad(X)
    P = -G_curr.copy()
    
    alpha = 0.1
    beta = 0.0
    start_time = time.time()
    best_E = E_curr
    best_X = X.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Line search
        best_alpha = alpha
        best_E_trial = E_curr
        best_X_trial = X
        best_G_trial = G_curr
        
        # Try a range of step sizes
        alphas = [alpha * 0.5, alpha, alpha * 1.5, alpha * 2.0, alpha * 0.25]
        for a in alphas:
            if a <= 0:
                continue
            X_trial = _retract(X, P, a)
            E_trial, G_trial = _energy_and_grad(X_trial)
            if E_trial < best_E_trial:
                best_E_trial = E_trial
                best_alpha = a
                best_X_trial = X_trial
                best_G_trial = G_trial
        
        if best_E_trial >= E_curr:
            # No improvement, reduce step size
            alpha *= 0.5
            if alpha < 1e-12:
                break
            continue
        
        # Update
        X = best_X_trial
        E_curr = best_E_trial
        G_curr = best_G_trial
        
        # Update best
        if E_curr < best_E:
            best_E = E_curr
            best_X = X.copy()
        
        # Convergence check
        if np.linalg.norm(G_curr) < 1e-10:
            break
        
        # Compute beta (Polak-Ribiere)
        P_old = P
        P = -G_curr + beta * P_old
        # Project to ensure P is tangent
        proj = np.sum(P * X, axis=1)
        P = P - proj[:, np.newaxis] * X
        
        # New beta
        if np.linalg.norm(P_old) > 1e-15:
            beta_new = np.sum(G_curr * (G_curr - P_old)) / np.linalg.norm(P_old)**2
            beta = max(0.0, beta_new)
        else:
            beta = 0.0
        
        # Update alpha strategy
        alpha = best_alpha * 0.9
        
    return best_X, best_E

def _lbfgs_optimize(X_init, time_budget, max_iter=2000):
    """
    L-BFGS-B optimization with normalization.
    """
    X = X_init.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-15)
    
    def objective(flat):
        X_arr = flat.reshape(N, 3)
        X_arr /= np.linalg.norm(X_arr, axis=1, keepdims=True)
        return _log_energy(X_arr)
    
    def gradient(flat):
        X_arr = flat.reshape(N, 3)
        X_arr /= np.linalg.norm(X_arr, axis=1, keepdims=True)
        E, G = _energy_and_grad(X_arr)
        return G.flatten()
    
    x0 = X.flatten()
    start_time = time.time()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'maxfun': 5000, 'gtol': 1e-8})
    
    X_opt = result.x.reshape(N, 3)
    X_opt /= np.linalg.norm(X_opt, axis=1, keepdims=True)
    E_opt = _log_energy(X_opt)
    
    return X_opt, E_opt

def _polish(X_init, time_budget):
    """
    High-precision polish using L-BFGS-B with tight tolerance.
    """
    X = X_init.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-15)
    
    def objective(flat):
        X_arr = flat.reshape(N, 3)
        X_arr /= np.linalg.norm(X_arr, axis=1, keepdims=True)
        return _log_energy(X_arr)
    
    def gradient(flat):
        X_arr = flat.reshape(N, 3)
        X_arr /= np.linalg.norm(X_arr, axis=1, keepdims=True)
        E, G = _energy_and_grad(X_arr)
        return G.flatten()
    
    x0 = X.flatten()
    start_time = time.time()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 3000, 'maxfun': 10000, 'gtol': 1e-10})
    
    X_opt = result.x.reshape(N, 3)
    X_opt /= np.linalg.norm(X_opt, axis=1, keepdims=True)
    E_opt = _log_energy(X_opt)
    
    return X_opt, E_opt

def _dual_optimize(X_init, time_budget):
    """
    Combined CG + L-BFGS optimization.
    """
    # First, use Riemannian CG for fast convergence
    X_cg, E_cg = _riemannian_cg(X_init, time_budget * 0.6)
    
    # Then, polish with L-BFGS-B
    remaining = time_budget - (time.time() - (time.time() - time_budget))  # This is wrong
    # Let's just use a fixed fraction
    X_final, E_final = _lbfgs_optimize(X_cg, time_budget * 0.4)
    
    return X_final, E_final

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral
    x_fib = _fibonacci_sphere()
    inits.append(x_fib)
    
    # 2. Halton sequence
    x_halton = _halton_sphere()
    inits.append(x_halton)
    
    # 3. Random
    x_rand = _random_sphere(42)
    inits.append(x_rand)
    
    # 4. Jittered Fibonacci
    rng = np.random.default_rng(123)
    x_jit = x_fib + rng.normal(scale=0.05, size=(N, 3))
    x_jit /= np.linalg.norm(x_jit, axis=1, keepdims=True)
    inits.append(x_jit)
    
    best_X = None
    best_E = float('inf')
    
    # Allocate time to each initialization
    # Total time ~ 145s
    # Let's give ~35s to each of 4 initializations
    
    for idx, X_init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
            
        budget = min(35.0, TIME_LIMIT - elapsed - 10)
        if budget <= 0:
            continue
            
        X_opt, E_opt = _dual_optimize(X_init, budget)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # Final polish?
    if best_X is not None and time.time() - start_time < TIME_LIMIT - 10:
        remaining = TIME_LIMIT - (time.time() - start_time) - 5
        if remaining > 2.0:
            X_polished, E_polished = _polish(best_X, remaining)
            if E_polished < best_E:
                best_X = X_polished
                best_E = E_polished

    if best_X is None:
        return x_fib
        
    # Ensure output is valid
    # Normalize again
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    
    # Check for invalid points
    for i in range(N):
        if not np.all(np.isfinite(best_X[i])):
            return x_fib
        if np.linalg.norm(best_X[i]) < 1e-9:
            return x_fib
    
    return best_X
