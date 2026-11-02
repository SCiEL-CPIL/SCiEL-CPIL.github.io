"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy(x):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(x)
    # Compute all pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dists[mask]
    if np.any(valid_dists < 1e-15):
        return 1e10
    return math.fsum(math.log(1.0 / d) for d in valid_dists)


def _log_energy_and_grad(x):
    """Compute logarithmic energy and its gradient on the sphere."""
    n = len(x)
    # Pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero
    eps = 1e-15
    dists = np.maximum(dists, eps)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dists[mask]))
    
    # Gradient: d/dx_i [-log(d_ij)] = (x_i - x_j) / d_ij^2 for j != i
    # So grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = ||x_i - x_j||^2
    dist2 = dists * dists
    dist2 = np.maximum(dist2, eps * eps)
    
    # grad[i] = sum_{j != i} (x_i - x_j) / dist2[i,j]
    # = sum_j (x_i - x_j)/dist2[i,j]  (with dist2[i,i] = inf so term is 0)
    np.fill_diagonal(dist2, np.inf)
    weights = 1.0 / dist2  # shape (n, n)
    # grad[i] = sum_j weights[i,j] * (x_i - x_j)
    # = sum_j weights[i,j]*x_i - sum_j weights[i,j]*x_j
    # = (sum_j weights[i,j]) * x_i - (weights @ x)[i]
    row_sums = weights.sum(axis=1)
    grad = row_sums[:, np.newaxis] * x - weights @ x
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . x) x
    dot = np.sum(grad * x, axis=1)
    grad_tangent = grad - dot[:, np.newaxis] * x
    
    return energy, grad_tangent


def _project_to_sphere(x):
    """Project points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def construct():
    t0 = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    # Try multiple random restarts with simulated annealing-like perturbation
    rng = np.random.default_rng(42)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    for restart in range(20):
        if time.time() - t0 > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far
            noise = rng.normal(0, 0.1, size=(N, 3))
            x_start = _project_to_sphere(x0 + noise)
        
        # Use L-BFGS-B with the tangent-space gradient
        # We parameterize in R^3 and project, but use gradient projected to tangent
        def fun(x_flat):
            x = _project_to_sphere(x_flat.reshape(N, 3))
            e, _ = _log_energy_and_grad(x)
            return e
        
        def jac(x_flat):
            x = _project_to_sphere(x_flat.reshape(N, 3))
            _, g = _log_energy_and_grad(x)
            return g.flatten()
        
        try:
            res = minimize(fun, x_start.flatten(), jac=jac, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            x_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() - t0 < deadline - 30:
        x_curr = best_x.copy()
        e_curr = best_e
        T = 0.5
        T_min = 1e-6
        cooling = 0.995
        
        while T > T_min and time.time() - t0 < deadline - 10:
            # Generate random perturbation on sphere
            noise = rng.normal(0, np.sqrt(T), size=(N, 3))
            x_trial = _project_to_sphere(x_curr + noise)
            
            e_trial = _log_energy(x_trial)
            delta_e = e_trial - e_curr
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                x_curr = x_trial
                e_curr = e_trial
                if e_curr < best_e:
                    best_e = e_curr
                    best_x = x_curr.copy()
            
            T *= cooling
        
        # Polish with L-BFGS-B from SA result
        if time.time() - t0 < deadline - 5:
            try:
                def fun(x_flat):
                    x = _project_to_sphere(x_flat.reshape(N, 3))
                    e, _ = _log_energy_and_grad(x)
                    return e
                
                def jac(x_flat):
                    x = _project_to_sphere(x_flat.reshape(N, 3))
                    _, g = _log_energy_and_grad(x)
                    return g.flatten()
                
                res = minimize(fun, x_curr.flatten(), jac=jac, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
                x_opt = _project_to_sphere(res.x.reshape(N, 3))
                e_opt = _log_energy(x_opt)
                
                if e_opt < best_e:
                    best_e = e_opt
                    best_x = x_opt.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() - t0 < deadline - 2:
        try:
            def fun(x_flat):
                x = _project_to_sphere(x_flat.reshape(N, 3))
                e, _ = _log_energy_and_grad(x)
                return e
            
            def jac(x_flat):
                x = _project_to_sphere(x_flat.reshape(N, 3))
                _, g = _log_energy_and_grad(x)
                return g.flatten()
            
            res = minimize(fun, best_x.flatten(), jac=jac, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            x_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _log_energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_x = _project_to_sphere(best_x)
    # Check for near-duplicate points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Nudge points apart
        rng2 = np.random.default_rng(123)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_x[i] - best_x[j])
                if d < 1e-9:
                    best_x[j] = _project_to_sphere(best_x[j] + rng2.normal(0, 0.01, size=3))
        best_x = _project_to_sphere(best_x)
    
    return best_x

# EVOLVE-BLOCK-END
