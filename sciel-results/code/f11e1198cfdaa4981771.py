"""Optimize 150 points on the unit sphere to minimize logarithmic energy."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(X):
    """Compute log energy and gradient for N points on the sphere.
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    where d_ij = ||x_i - x_j|| (chordal distance).
    
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    (for the unprojected energy; we project onto tangent space)
    """
    N = X.shape[0]
    # Pairwise distances
    # diff[i,j] = X[i] - X[j], shape (N, N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Energy: -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_upper = d[iu]
    # Clamp to avoid log(0)
    d_upper = np.maximum(d_upper, 1e-15)
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: d^2 = d2, and (x_i - x_j)/d^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j diff[i,j] / d2[i,j]
    # Set diagonal to 0
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(d2, 1.0)  # already done
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j inv_d2[i,j] * diff[i,j]
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    
    # Project gradient onto tangent space: grad_t = grad - (grad . x_i) x_i
    dot = np.sum(grad * X, axis=1, keepdims=True)
    grad = grad - dot * X
    
    return energy, grad

def _project_to_sphere(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X0.copy()
    best_energy = _energy_and_grad(X0)[0]
    
    # Use L-BFGS-B on flattened coordinates with sphere constraint handled by projection
    # We'll use a custom approach: optimize in R^{3N} and project after each step
    
    def objective_and_grad(flat_X):
        X = flat_X.reshape(N, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        # Gradient in flat space: project gradient back
        g_flat = g.flatten()
        return e, g_flat
    
    # Multi-restart with different initial conditions
    np.random.seed(42)
    
    # Try several initial configurations
    inits = [X0.copy()]
    
    # Add a few random perturbations
    for _ in range(2):
        X_rand = np.random.randn(N, 3)
        X_rand = _project_to_sphere(X_rand)
        inits.append(X_rand)
    
    # Halton sequence initialization
    def halton_sequence(n, base):
        seq = np.zeros(n)
        f = 1.0
        i = 0
        for j in range(n):
            i = j + 1
            f = 1.0 / base
            while i > 0:
                f /= base
                i = int(i / base)
                seq[j] += f * (i + 1) % base
            # Actually let me redo this properly
        return seq
    
    # Better Halton
    def halton(n, base):
        result = np.zeros(n)
        f = 1.0
        for i in range(n):
            i_val = i + 1
            f = 1.0 / base
            while i_val > 0:
                f /= base
                result[i] += f * (i_val % base)
                i_val //= base
        return result
    
    h1 = halton(N, 2)
    h2 = halton(N, 3)
    h3 = halton(N, 5)
    
    # Convert to 3D points on sphere using spherical coordinates
    theta = 2 * np.pi * h1
    cos_phi = 2 * h2 - 1
    sin_phi = np.sqrt(np.maximum(1 - cos_phi**2, 0))
    X_halton = np.column_stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
    X_halton = _project_to_sphere(X_halton)
    inits.append(X_halton)
    
    # Optimize each initial guess
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Flatten
        x0 = X_init.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            X_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _energy_and_grad(X_opt)[0]
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_X = X_opt.copy()
        except Exception:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 5.0:
        try:
            x0 = best_X.flatten()
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            X_polished = _project_to_sphere(res.x.reshape(N, 3))
            e_polished = _energy_and_grad(X_polished)[0]
            if e_polished < best_energy:
                best_energy = e_polished
                best_X = X_polished.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Validate: no points closer than 1e-9
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_X
