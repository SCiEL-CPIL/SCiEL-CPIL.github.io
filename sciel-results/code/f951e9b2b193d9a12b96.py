"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import log, fsum

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import log, fsum

def _energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    X: (N, 3) array of points on the unit sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    N = X.shape[0]
    # Compute pairwise distances
    # X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # For each i, sum over j != i of (X[i] - X[j]) / dist_sq[i,j]
    # diff[i,j] = X[i] - X[j]
    # grad[i] = sum_j (X[i] - X[j]) / dist_sq[i,j]
    
    # Create matrix where element (i,j) = (X[i]-X[j]) / dist_sq[i,j]
    # diff is (N, N, 3), dist_sq is (N, N)
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq[dist_sq > 1e-15] = 1.0 / dist_sq[dist_sq > 1e-15]
    
    # Weighted sum: grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _energy_only(X):
    """Just the energy, for scipy minimize."""
    X = _project_to_sphere(X)
    e, _ = _energy_and_grad(X)
    return e

def _energy_grad_only(X):
    """Energy and gradient for scipy minimize."""
    X = _project_to_sphere(X)
    return _energy_and_grad(X)

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure on sphere
    X = _project_to_sphere(X)
    
    best_X = X.copy()
    best_energy = _energy_only(X)
    
    # Use L-BFGS-B with projection via custom objective
    # Actually, let's use a different approach: parameterize on sphere or use 
    # gradient descent with projection
    
    # Approach: Gradient descent with projection to sphere
    # Step size schedule
    
    lr = 0.01
    patience = 0
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        X = _project_to_sphere(X)
        energy, grad = _energy_and_grad(X)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > 500:
            # Reduce learning rate
            lr *= 0.5
            patience = 0
            if lr < 1e-8:
                break
        
        # Gradient step
        X = X - lr * grad
        X = _project_to_sphere(X)
    
    # Try multiple random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initial points
        X = rng.standard_normal((N, 3))
        X = _project_to_sphere(X)
        
        lr = 0.01
        patience = 0
        
        for step in range(20000):
            if time.time() > deadline:
                break
            
            X = _project_to_sphere(X)
            energy, grad = _energy_and_grad(X)
            
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 300:
                lr *= 0.5
                patience = 0
                if lr < 1e-8:
                    break
            
            X = X - lr * grad
            X = _project_to_sphere(X)
    
    # Final polish with smaller steps
    if time.time() < deadline - 5:
        X = best_X.copy()
        X = _project_to_sphere(X)
        lr = 1e-6
        for step in range(10000):
            if time.time() > deadline:
                break
            X = _project_to_sphere(X)
            energy, grad = _energy_and_grad(X)
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
            X = X - lr * grad
            X = _project_to_sphere(X)
    
    return _project_to_sphere(best_X)
# EVOLVE-BLOCK-END
