"""
Minimize logarithmic energy of 100 points on the unit sphere.

Strategy:
1. Use multiple diverse initializations (Fibonacci spiral, Sobol, random).
2. Use SciPy's L-BFGS-B with Riemannian gradient (projected to tangent space).
3. The energy is E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
4. Riemannian gradient: project Euclidean gradient to tangent space.
5. Use multiple restarts and a final high-precision polish.
6. Add a Riemannian conjugate gradient method for faster convergence.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Returns (energy, gradient).
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i < j of -log(dist[i,j])
    triu_idx = np.triu_indices(N, k=1)
    d = dist[triu_idx]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Gradient: grad[i] = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _precise_energy(points):
    """Compute logarithmic energy with math.fsum for precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    triu_idx = np.triu_indices(N, k=1)
    d = dist[triu_idx]
    terms = [-math.log(d[k]) for k in range(len(d))]
    return math.fsum(terms)

def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _riemannian_gradient(points, grad_euclidean):
    """
    Project Euclidean gradient to tangent space of sphere at each point.
    Tangent space at x: {v : v . x = 0}
    Projection: v_tangent = v - (v . x) * x
    """
    dot = np.sum(grad_euclidean * points, axis=1, keepdims=True)
    return grad_euclidean - dot * points

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n=N, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _sobol_sphere(n=N):
    """Generate Sobol-like quasi-random points on the unit sphere."""
    pts = np.zeros((n, 3))
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        phi = 2.0 * np.pi * ((i * 0.6180339887498949) % 1.0)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        pts[i] = [r * np.cos(phi), r * np.sin(phi), z]
    return pts

def _optimize_lbfgs(points, deadline=None, max_iter=2000):
    """
    Optimize using L-BFGS-B with Riemannian gradient.
    Uses parameterization: points on sphere, gradient projected to tangent space.
    Returns (points, energy).
    """
    points = _project_to_sphere(points.copy())
    E = _precise_energy(points)
    
    def objective(x):
        """Objective function: energy of points on sphere."""
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        """Gradient function: Riemannian gradient projected to tangent space."""
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, grad_eucl = _energy_and_grad(pts)
        grad_riemannian = _riemannian_gradient(pts, grad_eucl)
        return grad_riemannian.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        
        pts = _project_to_sphere(result.x.reshape(N, 3))
        E_new = _precise_energy(pts)
        
        if E_new < E:
            return pts, E_new
    except Exception:
        pass
    
    return points, E

def _optimize_gradient_descent(points, max_iter=5000, lr_init=0.001, lr_decay=0.999,
                                min_lr=1e-6, deadline=None):
    """
    Riemannian gradient descent with adaptive learning rate.
    Returns (points, energy).
    """
    points = _project_to_sphere(points.copy())
    E = _precise_energy(points)
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad_eucl = _energy_and_grad(points)
        grad_riem = _riemannian_gradient(points, grad_eucl)
        
        grad_norm = np.linalg.norm(grad_riem)
        if grad_norm < 1e-14:
            break
        
        step = lr * grad_riem
        points_new = points + step
        points_new = _project_to_sphere(points_new)
        
        E_new = _precise_energy(points_new)
        
        if E_new < E:
            points = points_new
            E = E_new
            lr = min(lr * 1.02, 0.01)
        else:
            lr *= lr_decay
        
        if lr < min_lr:
            break
    
    return points, E

def _optimize_conjugate_gradient(points, max_iter=3000, deadline=None):
    """
    Riemannian conjugate gradient method for faster convergence.
    Uses Fletcher-Reeves restart rule.
    Returns (points, energy).
    """
    points = _project_to_sphere(points.copy())
    E = _precise_energy(points)
    
    # Initial gradient
    _, grad_eucl = _energy_and_grad(points)
    grad_riem = _riemannian_gradient(points, grad_eucl)
    d = -grad_riem  # Search direction
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Backtracking line search
        lr = 0.01
        for _ in range(20):
            points_trial = points + lr * d
            points_trial = _project_to_sphere(points_trial)
            E_trial = _precise_energy(points_trial)
            
            if E_trial < E:
                break
            lr *= 0.5
        
        if E_trial >= E:
            # Restart if line search fails
            d = -grad_riem
            continue
        
        # Update points
        points = points_trial
        E = E_trial
        
        # Compute new gradient
        _, grad_eucl_new = _energy_and_grad(points)
        grad_riem_new = _riemannian_gradient(points, grad_eucl_new)
        
        # Fletcher-Reeves beta
        grad_norm_sq = np.sum(grad_riem * grad_riem)
        if grad_norm_sq > 1e-30:
            beta = np.sum(grad_riem_new * grad_riem_new) / grad_norm_sq
        else:
            beta = 0.0
        
        # Update search direction
        d = -grad_riem_new + beta * d
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_riem_new)
        if grad_norm < 1e-12:
            break
    
    return points, E

def _polish(points, max_iter=10000, lr_init=0.0001, lr_decay=0.9999,
            min_lr=1e-8, deadline=None):
    """High-precision polish with very small steps."""
    points = _project_to_sphere(points.copy())
    E = _precise_energy(points)
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad_eucl = _energy_and_grad(points)
        grad_riem = _riemannian_gradient(points, grad_eucl)
        
        grad_norm = np.linalg.norm(grad_riem)
        if grad_norm < 1e-15:
            break
        
        step = lr * grad_riem
        points_new = points + step
        points_new = _project_to_sphere(points_new)
        
        E_new = _precise_energy(points_new)
        
        if E_new < E:
            points = points_new
            E = E_new
            lr = min(lr * 1.005, 0.001)
        else:
            lr *= lr_decay
        
        if lr < min_lr:
            break
    
    return points, E

def _local_search(points, deadline=None, max_iter=2000):
    """
    Local search with multiple methods: try gradient descent, conjugate gradient,
    and L-BFGS-B, keeping the best result.
    """
    best_points = points.copy()
    best_energy = _precise_energy(best_points)
    
    # Method 1: Conjugate gradient
    if deadline is not None and time.time() < deadline - 10.0:
        pts, E = _optimize_conjugate_gradient(best_points, max_iter=1000, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Method 2: L-BFGS-B
    if deadline is not None and time.time() < deadline - 5.0:
        pts, E = _optimize_lbfgs(best_points, deadline=deadline, max_iter=500)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Method 3: Gradient descent polish
    if deadline is not None and time.time() < deadline - 2.0:
        pts, E = _polish(best_points, max_iter=5000, lr_init=0.00005,
                         lr_decay=0.9999, min_lr=1e-9, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin for 150s limit
    
    best_points = _fibonacci_spiral(N)
    best_energy = _precise_energy(best_points)
    
    # Phase 1: Optimize from Fibonacci spiral
    if time.time() < deadline - 30.0:
        pts, E = _local_search(best_points, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Phase 2: Multiple restarts with diverse initializations
    initializations = [
        ("sobol", _sobol_sphere()),
        ("random1", _random_points(N, seed=1)),
        ("random2", _random_points(N, seed=2)),
        ("random3", _random_points(N, seed=3)),
        ("random4", _random_points(N, seed=4)),
        ("random5", _random_points(N, seed=5)),
    ]
    
    # Add perturbed versions of current best
    rng = np.random.default_rng(42)
    for i in range(4):
        noise = rng.standard_normal((N, 3)) * (0.05 + 0.08 * i)
        perturbed = _project_to_sphere(best_points + noise)
        initializations.append((f"perturbed_{i}", perturbed))
    
    for name, start_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        pts, E = _local_search(start_pts, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        pts, E = _polish(best_points, max_iter=10000, lr_init=0.00005,
                         lr_decay=0.9999, min_lr=1e-10, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Final verification: ensure all points are on unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Separate them
                rng = np.random.default_rng(i * 1000 + j)
                best_points[j] = best_points[i] + rng.standard_normal(3) * 0.01
                best_points = _project_to_sphere(best_points)
    
    return best_points
