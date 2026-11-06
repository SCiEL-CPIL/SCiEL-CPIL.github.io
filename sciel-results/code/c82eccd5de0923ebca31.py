"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dists = np.sqrt(dist_sq)
    np.fill_diagonal(dists, 1.0)
    
    # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    upper = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[upper]))
    
    # Gradient in ambient space:
    # grad_ambient_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-15
    inv_d2[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    # Efficient gradient computation
    # grad_ambient[i] = -x_i * sum_j inv_d2[i,j] + sum_j x_j * inv_d2[i,j]
    row_sums = np.sum(inv_d2, axis=1)
    term2 = (pts.T @ inv_d2).T
    grad_ambient = -pts * row_sums[:, np.newaxis] + term2
    
    # Project to tangent space
    dot_products = np.sum(grad_ambient * pts, axis=1)
    grad_tangent = grad_ambient - dot_products[:, np.newaxis] * pts
    
    return energy, grad_tangent

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _minimize_on_sphere(pts):
    """
    Minimize the logarithmic energy using L-BFGS-B on the sphere.
    We parameterize points on the sphere using a local chart (e.g., stereographic projection
    or simply optimize in R^3 with projection after each step).
    
    Here we use a simple approach: optimize in R^3 with L-BFGS-B, 
    but project back to sphere after each function evaluation.
    This is not perfectly constrained but works well in practice.
    
    Alternative: use scipy.optimize.minimize with method='L-BFGS-B' 
    on the flattened array, with a custom objective that projects to sphere.
    """
    # Flatten the points
    x0_flat = pts.flatten()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _project_to_sphere(x)
        energy, grad_tangent = _energy_and_grad(x)
        # We need to map the tangent gradient back to the ambient space for L-BFGS-B
        # Since we're projecting, the gradient in the flat space is the ambient gradient
        # projected appropriately. For simplicity, we use the ambient gradient.
        grad_ambient = grad_tangent + np.sum(grad_tangent * x, axis=1, keepdims=True) * x
        return energy, grad_ambient.flatten()
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_energy, _ = _energy_and_grad(best_pts)
    except Exception:
        best_pts = pts.copy()
        best_energy, _ = _energy_and_grad(best_pts)
    
    return best_pts, best_energy

def _gradient_descent_polish(pts, max_iters=10000, time_limit=30.0):
    """
    Final polish using gradient descent with adaptive step size.
    """
    x = pts.copy()
    best_x = x.copy()
    best_energy, _ = _energy_and_grad(x)
    
    start_time = time.time()
    lr = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(x)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
        
        # Backtracking line search
        alpha = 1.0
        step = -lr * grad
        improved = False
        for _ in range(30):
            x_trial = x + alpha * step
            x_trial = _project_to_sphere(x_trial)
            
            # Check for very close points
            diff = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            if np.min(dist_sq) < 1e-18:
                alpha *= 0.5
                continue
            
            e_trial, _ = _energy_and_grad(x_trial)
            
            if e_trial < energy - 1e-12 * alpha * np.sum(grad * grad):
                x = x_trial
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_x, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s kill
    
    # Initial point: Fibonacci sphere
    x0 = _fibonacci_sphere()
    e0, _ = _energy_and_grad(x0)
    best_pts = x0.copy()
    best_energy = e0
    
    # Phase 1: L-BFGS optimization from Fibonacci start
    elapsed = time.time() - start_time
    if elapsed < deadline - 40.0:
        x_opt, e_opt = _minimize_on_sphere(x0)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = x_opt.copy()
    
    # Phase 2: Try a few perturbed starts to escape local minima
    np.random.seed(42)
    for restart in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:
            break
        
        # Create a perturbed version of the best configuration
        noise_scale = 0.05 * (restart + 1)
        x_pert = best_pts + noise_scale * np.random.randn(N, 3)
        x_pert = _project_to_sphere(x_pert)
        
        x_opt, e_opt = _minimize_on_sphere(x_pert)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = x_opt.copy()
    
    # Phase 3: Final gradient descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        time_for_polish = min(30.0, deadline - 10.0 - elapsed)
        x_opt, e_opt = _gradient_descent_polish(best_pts, max_iters=10000, time_limit=time_for_polish)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = x_opt.copy()
    
    # Ensure valid output
    result = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    if np.min(dist_sq) < 1e-18:
        # Fallback to Fibonacci sphere
        result = x0.copy()
    
    # Verify finite values
    if not np.all(np.isfinite(result)):
        result = x0.copy()
    
    return result
# EVOLVE-BLOCK-END
