"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_grad(x):
    """x is (N,3) points on unit sphere. Returns energy and Riemannian gradient (N,3)."""
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)  # (N,N)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d_safe = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d_safe))
    
    # Riemannian gradient on sphere:
    # Euclidean gradient: grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Riemannian gradient = (I - x_i x_i^T) @ euclidean_grad_i
    # But since we're on the sphere and the function is rotationally invariant,
    # the Euclidean gradient is already tangent to the sphere (orthogonal to x_i)
    # because sum_j (x_i - x_j)/d_ij^2 dotted with x_i = sum_j (1 - x_i.x_j/d_ij^2)
    # which is NOT zero in general. So we need to project.
    
    D2 = dist * dist
    np.fill_diagonal(D2, 1.0)
    inv_D2 = 1.0 / D2
    inv_D2[np.diag_indices(N)] = 0.0
    
    # Euclidean gradient
    weighted_diff = diff * inv_D2[:, :, np.newaxis]
    eucl_grad = -np.sum(weighted_diff, axis=1)  # (N,3)
    
    # Project to tangent space: (I - x x^T) @ grad
    # x @ (x^T @ grad)
    x_dot_grad = np.sum(x * eucl_grad, axis=1, keepdims=True)  # (N,1)
    riem_grad = eucl_grad - x * x_dot_grad
    
    return energy, riem_grad


def _exponential_map(x, v, t=1.0):
    """Exponential map on unit sphere: move from x along tangent vector v by step t."""
    # v should be tangent to sphere at x
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    norm_v_safe = np.maximum(norm_v, 1e-15)
    angle = t * norm_v_safe  # (N,1)
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    # exp_map = cos(angle)*x + sin(angle)*v/norm_v
    v_normalized = v / norm_v_safe
    result = cos_angle * x + sin_angle * v_normalized
    # Handle where norm_v was tiny
    small = norm_v < 1e-15
    if np.any(small):
        result[small] = x[small]
    # Normalize to ensure on sphere
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-15)
    return result


def _riemannian_cg(x0, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Riemannian conjugate gradient on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    x = x0.copy()
    energy, grad = _energy_and_grad(x)
    
    # Initial direction: negative gradient
    p = -grad.copy()
    
    best_energy = energy
    best_x = x.copy()
    
    for k in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search: find step size t that minimizes energy along geodesic
        # Use backtracking line search
        t = 1.0
        c1 = 1e-4
        rho = 0.5
        max_ls = 30
        
        for ls_iter in range(max_ls):
            x_new = _exponential_map(x, p, t)
            energy_new, grad_new = _energy_and_grad(x_new)
            
            # Armijo condition
            if energy_new <= energy + c1 * t * np.sum(grad * p):
                break
            t *= rho
        
        # Update
        x = x_new
        energy = energy_new
        grad = grad_new
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
        
        # Update direction (Polak-Ribiere)
        if k > 0:
            beta = np.sum(grad * (grad - grad_prev)) / np.sum(grad_prev * grad_prev)
            beta = max(beta, 0)
            p = -grad + beta * p
        else:
            p = -grad
        
        # Ensure p is tangent
        x_dot_p = np.sum(x * p, axis=1, keepdims=True)
        p = p - x * x_dot_p
        
        grad_prev = grad.copy()
    
    return best_x, best_energy


def _polish_lbfgs(x0, time_limit=30.0, start_time=None):
    """Final polish using L-BFGS with penalty method for high precision."""
    if start_time is None:
        start_time = time.time()
    
    def objective_with_penalty(x_flat):
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty = 10000.0 * np.sum((norms - 1.0) ** 2)
        energy, grad = _energy_and_grad(pts)
        # Gradient of penalty
        penalty_grad = 20000.0 * (norms[:, np.newaxis] - 1.0) * pts / np.maximum(norms[:, np.newaxis], 1e-15)
        return energy + penalty, (grad + penalty_grad).flatten()
    
    try:
        result = minimize(
            objective_with_penalty,
            x0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        
        x_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        x_opt = x_opt / np.maximum(norms, 1e-15)
        
        true_energy, _ = _energy_and_grad(x_opt)
        return x_opt, true_energy
    except Exception:
        return x0, _energy_and_grad(x0)[0]


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit, keep 40s buffer)
    
    # Start from Fibonacci spiral
    x0 = _fibonacci_spiral(N)
    
    best_x = x0.copy()
    best_energy = _energy_and_grad(best_x)[0]
    
    np.random.seed(42)
    
    # Phase 1: Riemannian CG from multiple restarts
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        elif restart == 1:
            # Slightly perturbed Fibonacci
            noise = np.random.randn(N, 3) * 0.05
            x_perturbed = x0 + noise
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            x_start = x_perturbed / np.maximum(norms, 1e-15)
        elif restart == 2:
            # More perturbation
            noise = np.random.randn(N, 3) * 0.15
            x_perturbed = x0 + noise
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            x_start = x_perturbed / np.maximum(norms, 1e-15)
        else:
            # Random points on sphere
            x_start = np.random.randn(N, 3)
            norms = np.linalg.norm(x_start, axis=1, keepdims=True)
            x_start = x_start / np.maximum(norms, 1e-15)
        
        time_remaining = deadline - time.time() - 30.0  # Reserve 30s for polish
        if time_remaining < 5.0:
            break
        
        x_opt, energy_opt = _riemannian_cg(
            x_start, 
            max_iter=300, 
            tol=1e-10,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_x = x_opt.copy()
    
    # Phase 2: Polish the best result with L-BFGS
    if time.time() < deadline - 10.0:
        x_polished, energy_polished = _polish_lbfgs(
            best_x, 
            time_limit=20.0,
            start_time=start_time
        )
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_x = x_polished.copy()
    
    # Final normalization
    pts = best_x.copy()
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-15)
    
    return pts

# EVOLVE-BLOCK-END
