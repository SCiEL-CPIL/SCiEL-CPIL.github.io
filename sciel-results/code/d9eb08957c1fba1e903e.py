"""Combined program for 100-point logarithmic energy minimization on the unit sphere.
Uses Riemannian gradient flow for fast convergence, L-BFGS-B for high-precision polish,
and simulated annealing for escaping local minima.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision (exact scoring)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_fast(points):
    """Fast energy (np.sum) and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent


def _project(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7, patience=2000):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_fast(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_fast(best_points)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(25):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_grad_fast(new_points)
            
            if new_energy < energy - 1e-10:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points


def _lbfgs_polish(points, time_budget, maxiter=10000):
    """High-precision L-BFGS-B polish in ambient space with projection."""
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_fast(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _simulated_annealing_refine(points, time_budget):
    """Refine points using simulated annealing-like perturbation."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy_fsum(best_points)
    T = 0.1
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi) * T
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        c = math.cos(angle)
        s = math.sin(angle)
        k = axis
        k_outer_k = np.outer(k, k)
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
        
        new_points = best_points.copy()
        new_points[idx] = R @ new_points[idx]
        new_points = _project(new_points)
        
        new_energy = _energy_fsum(new_points)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            best_points = new_points
            best_energy = new_energy
        
        T *= 0.995
    
    return best_points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project(points)


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy_fsum(best_points)
    
    # Phase 1: Riemannian gradient flow from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.3:
        budget1 = min(TIME_LIMIT * 0.25, 20.0)
        refined = _riemannian_flow(points, budget1, initial_lr=0.05)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: L-BFGS-B polish on the Riemannian result
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.5:
        budget2 = min(TIME_LIMIT * 0.2, 15.0)
        polished = _lbfgs_polish(best_points, budget2, maxiter=5000)
        e = _energy_fsum(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Phase 3: Multiple random restarts with Riemannian flow + quick polish
    np.random.seed(123)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.75:
            break
        restart_count += 1
        
        random_points = _random_points(N)
        
        budget_per_restart = min(TIME_LIMIT * 0.03, (TIME_LIMIT * 0.75 - elapsed) * 0.8)
        if budget_per_restart < 1.0:
            break
        
        # Quick Riemannian flow
        refined = _riemannian_flow(random_points, budget_per_restart * 0.6, initial_lr=0.1, patience=500)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
            # Quick polish on the improved result
            if time.time() - start_time < TIME_LIMIT * 0.75:
                polished = _lbfgs_polish(best_points, min(3.0, (TIME_LIMIT * 0.75 - (time.time() - start_time)) * 0.5), maxiter=2000)
                e2 = _energy_fsum(polished)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = polished
    
    # Phase 4: Simulated annealing refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.85:
        remaining = TIME_LIMIT * 0.85 - elapsed
        if remaining > 2.0:
            np.random.seed(456)
            sa_points = _simulated_annealing_refine(best_points, remaining * 0.7)
            e = _energy_fsum(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: Final high-precision L-BFGS-B polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.95:
        remaining = TIME_LIMIT * 0.95 - elapsed
        if remaining > 3.0:
            refined = _lbfgs_polish(best_points, remaining * 0.8, maxiter=10000)
            e = _energy_fsum(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Final projection and validation
    best_points = _project(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy_fsum(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
