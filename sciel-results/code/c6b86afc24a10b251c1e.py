import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad_cartesian(X, lambda_pen=10000.0):
    """
    Calculate energy and gradient for points X in R^3, with a penalty for deviating from the unit sphere.
    X: (N, 3) array
    Returns: energy (float), grad (N, 3) array
    """
    N = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist[i,j])
    iu = np.triu_indices(N, 1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient for point k: sum_{j != k} (X[k] - X[j]) / dist[k, j]^2 is the descent direction.
    # True gradient of E_log is - sum (X[k] - X[j]) / dist[k, j]^2
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # descent_dir[i] = sum_j diff[i, j] * inv_dist_sq[i, j]
    descent_dir = (diff * inv_dist_sq[:, :, np.newaxis]).sum(axis=1)
    grad_log = -descent_dir
    
    # Penalty: P = lambda * sum_i (||x_i||^2 - 1)^2
    # dP/dx_i = lambda * 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * lambda * (||x_i||^2 - 1) * x_i
    norms_sq = np.sum(X * X, axis=1)
    penalty_energy = lambda_pen * np.sum((norms_sq - 1.0) ** 2)
    grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * X
    
    total_energy = energy + penalty_energy
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms

def _precise_energy(X):
    """Calculate the precise logarithmic energy using math.fsum as per the task specification."""
    N = X.shape[0]
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _minimize_energy_lbfgs(X0, time_limit=30, ftol=1e-12, gtol=1e-12):
    """
    Minimize energy using L-BFGS-B on Cartesian coordinates with penalty.
    """
    X0_flat = X0.flatten()
    
    def fun(x):
        X = x.reshape(N, 3)
        e, _ = _energy_and_grad_cartesian(X)
        return e
    
    def jac(x):
        X = x.reshape(N, 3)
        _, g = _energy_and_grad_cartesian(X)
        return g.flatten()
        
    start_time = time.time()
    
    # Estimate max iterations based on time limit
    # Each iteration takes roughly 5-20ms for N=100, so ~50-200 iterations/sec
    # Be conservative: assume 100 iterations/sec
    estimated_iters = int(time_limit * 80)
    estimated_iters = max(estimated_iters, 500)
    estimated_iters = min(estimated_iters, 5000)
    
    res = minimize(fun, X0_flat, jac=jac, method='L-BFGS-B', 
                   options={'maxiter': estimated_iters, 'ftol': ftol, 'gtol': gtol})
    
    X_opt = res.x.reshape(N, 3)
    X_opt = _project_to_sphere(X_opt)
    
    return X_opt, res.fun

def construct():
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin for the 150s kill limit
    
    best_X = None
    best_E = float('inf')
    
    # Initialization 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 30:
        X_opt, E_opt = _minimize_energy_lbfgs(X_fib, time_limit=25)
        X_opt = _project_to_sphere(X_opt)
        e_true = _precise_energy(X_opt)
        if e_true < best_E:
            best_E = e_true
            best_X = X_opt

    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 30:
        # Initialization 2: Random
        rng = np.random.default_rng(42)
        X_rand = rng.normal(size=(N, 3))
        X_rand = _project_to_sphere(X_rand)
        
        X_opt, E_opt = _minimize_energy_lbfgs(X_rand, time_limit=25)
        X_opt = _project_to_sphere(X_opt)
        e_true = _precise_energy(X_opt)
        if e_true < best_E:
            best_E = e_true
            best_X = X_opt

    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 30:
        # Initialization 3: Sobol
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=3, scramble=True)
            X_sobol = sampler.random(N)
            X_sobol = _project_to_sphere(X_sobol)
            
            X_opt, E_opt = _minimize_energy_lbfgs(X_sobol, time_limit=25)
            X_opt = _project_to_sphere(X_opt)
            e_true = _precise_energy(X_opt)
            if e_true < best_E:
                best_E = e_true
                best_X = X_opt
        except ImportError:
            pass

    if best_X is not None:
        # Final high-precision polish with tighter tolerances
        elapsed = time.time() - start_time
        remaining_time = total_time_limit - elapsed
        if remaining_time > 10:
            X_polish, E_polish = _minimize_energy_lbfgs(best_X, time_limit=remaining_time - 3, 
                                                         ftol=1e-14, gtol=1e-14)
            X_polish = _project_to_sphere(X_polish)
            e_true = _precise_energy(X_polish)
            if e_true < best_E:
                best_E = e_true
                best_X = X_polish
    
    return best_X
