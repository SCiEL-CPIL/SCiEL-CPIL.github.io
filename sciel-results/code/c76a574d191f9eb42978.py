"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
import random

N = 100
DEADLINE = 145.0  # Safety margin before 150s limit


def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return math.fsum(math.log(1.0 / d) for d in dist[iu])


def _energy_fast(points):
    """Fast energy computation for optimization (not final scoring)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return np.sum(np.log(1.0 / dist[iu]))


def _energy_grad(points):
    """Compute the tangent-space gradient of the logarithmic energy."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot * points
    
    return grad_tangent


def _fibonacci_spiral(n, offset=0.5):
    """Generate n points on the unit sphere using the golden angle spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _icosahedron_sphere():
    """Generate points based on icosahedron vertices with subdivisions."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add more points using Fibonacci spiral as base
    return _fibonacci_spiral(N)


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _optimize_lbfgs(points, maxiter=2000, ftol=1e-14, gtol=1e-12, deadline=None):
    """Run L-BFGS-B optimization with tangent-space gradient."""
    from scipy.optimize import minimize
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _energy_fast(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _energy_grad(pts)
        return g.flatten()
    
    result = minimize(
        objective,
        points.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    if deadline is not None and time.time() > deadline:
        return points, _energy_fast(points)
    
    new_points = _project_to_sphere(result.x.reshape(N, 3))
    return new_points, _energy_fast(new_points)


def _nesterov_accelerated(points, steps=500, lr0=0.05, deadline=None):
    """Nesterov accelerated gradient descent on the sphere."""
    pts = points.copy()
    v = np.zeros_like(pts)
    lr = lr0
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Nesterov lookahead
        pts_next = pts - lr * v
        pts_next = _project_to_sphere(pts_next)
        
        g = _energy_grad(pts_next)
        v = 0.9 * v + g
        pts = pts_next - lr * v
        pts = _project_to_sphere(pts)
        
        lr = lr0 / (1.0 + step * 0.01)
    
    return pts, _energy_fast(pts)


def _molecular_dynamics(points, steps=200, deadline=None):
    """Simple MD-like approach with temperature annealing."""
    pts = points.copy()
    v = np.zeros_like(pts)
    temp = 0.01
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        g = _energy_grad(pts)
        v = 0.9 * v - 0.01 * g
        v += temp * np.random.randn(*pts.shape)
        pts = pts + v
        pts = _project_to_sphere(pts)
        
        temp *= 0.99
    
    return pts, _energy_fast(pts)


def _polish_ultra_precise(points, maxiter=5000, ftol=1e-16, gtol=1e-14, deadline=None):
    """Final high-precision polish with very tight tolerances."""
    from scipy.optimize import minimize
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _energy_fast(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _energy_grad(pts)
        return g.flatten()
    
    result = minimize(
        objective,
        points.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    if deadline is not None and time.time() > deadline:
        return points, _energy_fast(points)
    
    new_points = _project_to_sphere(result.x.reshape(N, 3))
    return new_points, _energy_fast(new_points)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_remaining():
        return deadline - time.time()
    
    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N, 0.5))
    initial_configs.append(_fibonacci_spiral(N, 0.3))
    initial_configs.append(_fibonacci_spiral(N, 0.7))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        initial_configs.append(_random_sphere(N, seed=seed))
    
    # Icosahedron-based
    initial_configs.append(_icosahedron_sphere())
    
    # Phase 2: Optimize each configuration with L-BFGS
    best_points = None
    best_energy = float('inf')
    
    for i, config in enumerate(initial_configs):
        if time_remaining() < 10.0:
            break
        
        pts, energy = _optimize_lbfgs(config, maxiter=1000, ftol=1e-12, gtol=1e-10, deadline=deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
        
        # If we have time, also try Nesterov on the best so far
        if time_remaining() > 30.0 and i % 2 == 0:
            pts_n, energy_n = _nesterov_accelerated(config, steps=300, lr0=0.03, deadline=deadline)
            if energy_n < best_energy:
                best_energy = energy_n
                best_points = pts_n.copy()
    
    # Phase 3: Take the best and do a more thorough optimization
    if best_points is not None and time_remaining() > 20.0:
        pts, energy = _optimize_lbfgs(best_points, maxiter=2000, ftol=1e-13, gtol=1e-11, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Phase 4: Try MD from the best point if time permits
    if best_points is not None and time_remaining() > 25.0:
        pts_md, energy_md = _molecular_dynamics(best_points, steps=100, deadline=deadline)
        if energy_md < best_energy:
            best_energy = energy_md
            best_points = pts_md.copy()
            # Re-optimize from MD result
            if time_remaining() > 15.0:
                pts, energy = _optimize_lbfgs(best_points, maxiter=1500, ftol=1e-12, gtol=1e-10, deadline=deadline)
                if energy < best_energy:
                    best_energy = energy
                    best_points = pts.copy()
    
    # Phase 5: Multiple restarts from perturbed best point
    if best_points is not None and time_remaining() > 40.0:
        for restart in range(5):
            if time_remaining() < 15.0:
                break
            
            # Perturb the best point
            noise_scale = 0.05 / (restart + 1)
            perturbed = best_points + noise_scale * np.random.randn(N, 3)
            perturbed = _project_to_sphere(perturbed)
            
            pts, energy = _optimize_lbfgs(perturbed, maxiter=1500, ftol=1e-12, gtol=1e-10, deadline=deadline)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
    
    # Phase 6: Final ultra-precise polish
    if best_points is not None and time_remaining() > 10.0:
        pts, energy = _polish_ultra_precise(best_points, maxiter=5000, ftol=1e-15, gtol=1e-13, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for any invalid points
    for i in range(N):
        if not np.all(np.isfinite(best_points[i])):
            best_points[i] = np.array([0, 0, 1])
        norm = np.linalg.norm(best_points[i])
        if norm < 1e-15:
            best_points[i] = np.array([0, 0, 1])
        else:
            best_points[i] /= norm
    
    return best_points
