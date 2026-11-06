"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|) and its gradient.
    pts: (N,3) array on the unit sphere.
    Returns (energy, grad) where grad is (N,3).
    """
    # Gram matrix
    G = pts @ pts.T  # (N, N)
    # Distances squared: d^2 = 2 - 2*G
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    
    # Energy: sum_{i<j} log(1/sqrt(d2)) = -0.5 * sum_{i<j} log(d2)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d2_upper = d2[iu]
    
    # For numerical stability, clip very small distances
    d2_safe = np.maximum(d2_upper, 1e-18)
    energy = -0.5 * np.sum(np.log(d2_safe))
    
    # Gradient:
    # dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    # Because d/dxi log(1/|xi-xj|) = -d/dxi log|xi-xj| = -(xi-xj)/|xi-xj|^2
    # Wait: E = sum log(1/d) = -sum log(d)
    # dE/dxi = -sum_{j != i} d/dxi log|xi - xj| = -sum_{j != i} (xi - xj)/|xi - xj|^2
    # So grad_i = -sum_{j != i} (xi - xj) / d2_ij
    
    # Let's compute this efficiently
    # grad_i = -sum_{j != i} (xi - xj) / d2_ij
    # = -[sum_j (xi - xj)/d2_ij] + (xi - xi)/d2_ii
    # = -[xi * sum_j 1/d2_ij - sum_j xj / d2_ij]
    
    # Build weight matrix W where W_ij = 1/d2_ij for i != j, 0 otherwise
    W = np.zeros_like(d2)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / d2_safe  # but we need to handle the full matrix, not just upper
    # Actually d2 is symmetric, so let's recompute full d2 for weights
    d2_full = 2.0 - 2.0 * G
    np.fill_diagonal(d2_full, 1.0)
    d2_full = np.maximum(d2_full, 1e-18)
    W = 1.0 / d2_full
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -[xi * sum_j W_ij - sum_j W_ij * xj]
    # = -[xi * row_sum_i - (W @ pts)_i]
    row_sums = W.sum(axis=1)  # (N,)
    W_pts = W @ pts  # (N, 3)
    grad = -(pts * row_sums[:, np.newaxis] - W_pts)
    
    # Project gradient to tangent space: grad_tan = grad - (grad . xi) xi
    # Since pts are on sphere, xi . xi = 1
    grad_proj = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
    
    return energy, grad_proj


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lbfgsb_optimize(pts, maxiter=1000, tol=1e-10):
    """
    Run L-BFGS-B optimization on the sphere using the projected gradient.
    We optimize in R^3 and project back to the sphere after each step.
    """
    def objective_and_grad(x):
        pts = _project_to_sphere(x)
        energy, grad = _energy_and_grad(pts)
        return energy, grad
    
    # Flatten
    x0 = pts.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxls': 50})
    pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    energy_opt = result.fun
    return pts_opt, energy_opt


def _riemannian_gd_polish(pts, lr=0.001, maxiter=5000, tol=1e-12):
    """
    Riemannian gradient descent polish on the sphere.
    """
    current_pts = pts.copy()
    current_energy = _energy_and_grad(current_pts)[0]
    
    for it in range(maxiter):
        energy, grad = _energy_and_grad(current_pts)
        
        # Check convergence
        if abs(current_energy - energy) < tol and it > 100:
            break
        current_energy = energy
        
        # Gradient step
        step = lr * grad
        
        # Move along the tangent direction using exponential map
        # For small steps, exp_map(v) ≈ x + v - 0.5*(x·v)*x
        # More accurately: if v is tangent, exp_x(v) = x*cos(|v|) + (v/|v|)*sin(|v|)
        # For simplicity and robustness, use a first-order approximation with projection
        new_pts = current_pts + step
        new_pts = _project_to_sphere(new_pts)
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(20):
            trial_pts = _project_to_sphere(current_pts + alpha * step)
            trial_energy = _energy_and_grad(trial_pts)[0]
            if trial_energy < energy:
                new_pts = trial_pts
                break
            alpha *= 0.5
        else:
            # If no improvement, accept the original step
            new_pts = _project_to_sphere(current_pts + step)
        
        # Check for too-close points
        G = new_pts @ new_pts.T
        d2 = 2.0 - 2.0 * G
        np.fill_diagonal(d2, 1.0)
        if d2.min() < 1e-18:
            # Points too close, reduce step
            new_pts = _project_to_sphere(current_pts + 0.1 * step)
        
        current_pts = new_pts
        
        # Reduce learning rate slowly
        if it % 500 == 0 and it > 0:
            lr *= 0.95
    
    final_energy = _energy_and_grad(current_pts)[0]
    return current_pts, final_energy


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Leave 20s margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere()
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Choose initialization
        if restarts == 1:
            start_pts = _fibonacci_sphere()
        elif restarts == 2:
            start_pts = _random_start(0)
        elif restarts == 3:
            start_pts = _random_start(1)
        else:
            # Perturb the best solution found so far
            noise = rng.standard_normal((N, 3)) * 0.1
            start_pts = _project_to_sphere(best_pts + noise)
        
        # L-BFGS optimization
        pts_opt, energy_opt = _lbfgsb_optimize(start_pts, maxiter=500, tol=1e-8)
        
        # Check validity
        G = pts_opt @ pts_opt.T
        d2 = 2.0 - 2.0 * G
        np.fill_diagonal(d2, 1.0)
        if d2.min() < 1e-18 or not np.all(np.isfinite(pts_opt)):
            continue
        
        # Riemannian GD polish
        pts_polished, energy_polished = _riemannian_gd_polish(pts_opt, lr=0.005, maxiter=3000, tol=1e-10)
        
        # Final check
        G = pts_polished @ pts_polished.T
        d2 = 2.0 - 2.0 * G
        np.fill_diagonal(d2, 1.0)
        if d2.min() < 1e-18 or not np.all(np.isfinite(pts_polished)):
            continue
        
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pts = pts_polished.copy()
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        best_pts, best_energy = _riemannian_gd_polish(best_pts, lr=0.001, maxiter=10000, tol=1e-12)
        G = best_pts @ best_pts.T
        d2 = 2.0 - 2.0 * G
        np.fill_diagonal(d2, 1.0)
        if d2.min() < 1e-18 or not np.all(np.isfinite(best_pts)):
            # Fallback to previous best
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere()
    
    return best_pts
# EVOLVE-BLOCK-END
