"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy_and_grad(X):
    """Compute log energy and gradient. X is (N,3) on unit sphere.
    Energy = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # Pairwise distances
    # ||x_i - x_j||^2 = 2 - 2*x_i·x_j
    G = X @ X.T
    np.fill_diagonal(G, 0.0)
    d2 = 2.0 - 2.0 * G  # squared chordal distances
    # Avoid log(0)
    d2 = np.maximum(d2, 1e-30)
    
    # Energy: -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # = -0.5 * sum_{i!=j} 0.5 * log(d2_ij) = -0.25 * sum_{i!=j} log(d2_ij)
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    energy = -0.25 * np.sum(log_d2)
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # = sum_{j!=i} (x_i - x_j) / d2_ij
    # Build matrix: for each pair (i,j), contribution to grad_i is (x_i - x_j)/d2_ij
    # grad = X * (sum_j 1/d2_ij) - sum_j x_j / d2_ij
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    row_sum = np.sum(inv_d2, axis=1)  # sum_j 1/d2_ij for each i
    # sum_j x_j / d2_ij = inv_d2 @ X
    weighted_sum = inv_d2 @ X
    grad = X * row_sum[:, np.newaxis] - weighted_sum
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return X / norms


def _riemannian_cg(X, max_iter=5000, tol=1e-12):
    """Riemannian conjugate gradient on the sphere."""
    X = _project_to_sphere(X.copy())
    energy, grad = _log_energy_and_grad(X)
    
    # Riemannian gradient = projection of Euclidean gradient
    # For sphere: rgrad = grad - (grad · x) * x
    def rgrad(X, grad):
        proj = np.sum(grad * X, axis=1, keepdims=True)
        return grad - proj * X
    
    rgrad = _rgrad(X, grad)
    
    # Initial search direction
    d = -rgrad.copy()
    prev_rgrad = rgrad.copy()
    prev_d = d.copy()
    
    for it in range(max_iter):
        if np.linalg.norm(rgrad) < tol:
            break
        
        # Line search: minimize f(t) = energy(retract(X, t*d))
        # Retraction on sphere: retract(X, d) = normalize(X + d)
        # We use backtracking line search
        
        alpha = 0.1
        c1 = 1e-4  # Armijo condition
        
        # Compute gradient direction
        # For Riemannian CG, we need the directional derivative
        # d/dt energy(retract(X, t*d)) at t=0 = <rgrad, d>
        dir_deriv = np.sum(rgrad * d)
        
        if dir_deriv >= 0:
            # Reset direction (restart)
            d = -rgrad.copy()
            dir_deriv = np.sum(rgrad * d)
        
        for ls_iter in range(50):
            X_new = _project_to_sphere(X + alpha * d)
            e_new, g_new = _log_energy_and_grad(X_new)
            
            # Armijo condition: e_new <= e + c1 * alpha * dir_deriv
            if e_new <= energy + c1 * alpha * dir_deriv:
                break
            alpha *= 0.5
        
        # Update
        X = X_new
        energy = e_new
        grad = g_new
        rgrad_new = _rgrad(X, grad)
        
        if np.linalg.norm(rgrad_new) < tol:
            X = _project_to_sphere(X)
            break
        
        # Fletcher-Reeves beta
        rgrad_norm_sq = np.sum(rgrad_new * rgrad_new)
        prev_rgrad_norm_sq = np.sum(prev_rgrad * prev_rgrad)
        if prev_rgrad_norm_sq > 1e-30:
            beta = rgrad_norm_sq / prev_rgrad_norm_sq
        else:
            beta = 0.0
        
        # Ensure beta is non-negative
        beta = max(beta, 0.0)
        
        d = -rgrad_new + beta * prev_d
        prev_rgrad = rgrad_new
        prev_d = d
        rgrad = rgrad_new
    
    return _project_to_sphere(X), energy


def _rgrad(X, grad):
    """Riemannian gradient on sphere."""
    proj = np.sum(grad * X, axis=1, keepdims=True)
    return grad - proj * X


def _newton_polish(X, max_iter=50, tol=1e-14):
    """Newton-type polish using approximate Hessian."""
    X = _project_to_sphere(X.copy())
    energy, grad = _log_energy_and_grad(X)
    rgrad = _rgrad(X, grad)
    
    for it in range(max_iter):
        if np.linalg.norm(rgrad) < tol:
            break
        
        # Approximate Hessian via finite differences of Riemannian gradient
        # Use a small step for Hessian approximation
        h = 1e-5
        H_approx = np.zeros((N * 3, N * 3))
        
        # This is expensive; use a simpler approach: trust region with BFGS-like update
        # Instead, use a damped Newton step with line search
        
        # Simple approach: take a step in -rgrad direction with adaptive step size
        step_size = 0.1
        for ls in range(30):
            X_new = _project_to_sphere(X + step_size * (-rgrad))
            e_new, g_new = _log_energy_and_grad(X_new)
            rgrad_new = _rgrad(X_new, g_new)
            
            if e_new < energy:
                X = X_new
                energy = e_new
                grad = g_new
                rgrad = rgrad_new
                break
            step_size *= 0.5
        
        else:
            break
    
    return _project_to_sphere(X), energy


def construct():
    """Construct 150 points on the unit sphere minimizing log energy."""
    start_time = time.time()
    deadline = start_time + 125.0  # 5 second safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _project_to_sphere(X)
    
    best_X = X.copy()
    best_energy = _log_energy_and_grad(X)[0]
    
    # Stage 1: Riemannian CG
    X1, e1 = _riemannian_cg(X, max_iter=3000, tol=1e-10)
    if e1 < best_energy:
        best_X = X1.copy()
        best_energy = e1
    
    if time.time() > deadline - 30:
        return _project_to_sphere(best_X)
    
    # Stage 2: More CG iterations with tighter tolerance
    X2, e2 = _riemannian_cg(best_X, max_iter=5000, tol=1e-12)
    if e2 < best_energy:
        best_X = X2.copy()
        best_energy = e2
    
    if time.time() > deadline - 10:
        return _project_to_sphere(best_X)
    
    # Stage 3: Newton polish
    X3, e3 = _newton_polish(best_X, max_iter=50, tol=1e-14)
    if e3 < best_energy:
        best_X = X3.copy()
        best_energy = e3
    
    # Final validation
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-15):
        # Fallback to initial
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END

