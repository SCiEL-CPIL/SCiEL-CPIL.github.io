"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i - x_j)/|x_i-x_j| = -(x_i-x_j)/d^2
            # Actually: log(1/d) = -log(d), d = |x_i - x_j|
            # d/dx_i (-log(d)) = -(1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
            term = -diff / (d * d)
            grad[i] += term
            grad[j] -= term
    return grad


def _project_to_sphere(points):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_random():
    """Generate random points on the sphere."""
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize_from(points, method='L-BFGS-B', maxiter=5000):
    """Optimize points using gradient-based methods with sphere projection."""
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _log_energy_grad(pts)
        # Project gradient to tangent space
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()
    
    x0 = points.flatten()
    res = minimize(objective, x0, jac=gradient, method=method,
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = _project_to_sphere(res.x.reshape(N, 3))
    return pts, res.fun


def _local_refinement(points, n_iter=2000, step_scale=0.1):
    """Perform local refinement using projected gradient descent."""
    pts = points.copy()
    for it in range(n_iter):
        grad = _log_energy_grad(pts)
        # Project to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        # Adaptive step size
        step = step_scale / (1.0 + 0.01 * it)
        new_pts = pts - step * grad
        # Project to sphere
        new_pts = _project_to_sphere(new_pts)
        # Check energy improvement
        old_e = _log_energy(pts)
        new_e = _log_energy(new_pts)
        if new_e < old_e:
            pts = new_pts
        else:
            step_scale *= 0.9
        if step_scale < 1e-8:
            break
    return pts


def construct():
    # Generate multiple initial configurations
    init_fib = _initial_fibonacci()
    init_rand = _initial_random()
    
    # Try multiple optimization strategies
    best_pts = None
    best_energy = float('inf')
    
    candidates = [init_fib, init_rand]
    
    for init in candidates:
        # L-BFGS-B optimization
        try:
            pts, energy = _optimize_from(init, method='L-BFGS-B', maxiter=3000)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
        
        # Local refinement
        try:
            pts_ref = _local_refinement(pts, n_iter=1000)
            e_ref = _log_energy(pts_ref)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref.copy()
        except Exception:
            pass
    
    # Additional refinement with smaller steps
    if best_pts is not None:
        try:
            final_pts = _local_refinement(best_pts, n_iter=500, step_scale=0.05)
            e_final = _log_energy(final_pts)
            if e_final < best_energy:
                best_pts = final_pts
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb
                best_pts[j] += 1e-6 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
