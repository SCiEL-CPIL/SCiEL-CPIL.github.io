"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # penalty for overlapping points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _gradient_energy(points):
    """Compute gradient of logarithmic energy w.r.t. each point."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
            # Actually: log(1/d) = -log(d), so d/dx_i (-log(d)) = -(1/d) * d/dx_i d
            # d/dx_i |x_i - x_j| = (x_i - x_j)/|x_i - x_j|
            # So gradient contribution to point i: -(1/d) * (x_i - x_j)/d = -(x_i - x_j)/d^2
            # Gradient contribution to point j: +(1/d) * (x_i - x_j)/d = +(x_i - x_j)/d^2
            term = diff / (d * d)
            grad[i] -= term
            grad[j] += term
    return grad


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def construct():
    # Start with a Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use L-BFGS-B with projected gradient approach
    # We'll use a parameterization that keeps points on the sphere by working in R^3
    # and projecting after each step, or use a custom approach.
    
    # Alternative: use scipy minimize with a custom gradient and constraints
    # Since L-BFGS-B doesn't handle equality constraints well, we'll use
    # the approach of minimizing in ambient space and projecting.
    
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    # Try L-BFGS-B with line search that projects
    def objective_and_grad(flat_points):
        pts = flat_points.reshape(N, 3)
        # Project to sphere
        pts = _project_to_sphere(pts)
        energy = _log_energy(pts)
        grad = _gradient_energy(pts)
        # The gradient in the ambient space needs adjustment for the projection,
        # but as an approximation, we can use the projected gradient
        return energy, grad.flatten()
    
    # Use multiple restarts with different initial configurations
    np.random.seed(42)
    
    for restart in range(5):
        if restart == 0:
            current = points.copy()
        else:
            # Perturb the current best
            current = best_points + np.random.normal(0, 0.01, size=(N, 3))
            current = _project_to_sphere(current)
        
        # Optimize using L-BFGS-B
        def fun(flat_x):
            e, g = objective_and_grad(flat_x)
            return e
        
        def jac(flat_x):
            e, g = objective_and_grad(flat_x)
            return g
        
        result = minimize(
            fun,
            current.flatten(),
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        
        opt_points = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_points)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
    
    # Final local refinement with smaller steps
    for _ in range(1000):
        grad = _gradient_energy(best_points)
        # Gradient descent step on the sphere (project gradient to tangent space)
        # Project gradient to tangent space: g_tangent = g - (g·x)x
        g_tangent = grad - np.sum(grad * best_points, axis=1, keepdims=True) * best_points
        lr = 1e-4
        new_points = best_points - lr * g_tangent
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points
        else:
            break
    
    return best_points
# EVOLVE-BLOCK-END
