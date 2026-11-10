"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points (N,3)."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                continue
            # d/dx_i log(1/d) = -1/d^2 * (x_i - x_j) / d = -(x_i - x_j)/d^2 ... 
            # Actually: E = sum log(1/d_ij) = -sum log(d_ij)
            # d/dx_i (-log d_ij) = -(1/d_ij) * d_ij / d_ij^2 ... 
            # d/dx_i d_ij = (x_i - x_j)/d_ij
            # d/dx_i (-log d_ij) = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
            coeff = -1.0 / (d * d)
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    min_lr = 1e-8
    patience = 0
    max_patience = 200
    
    while time.time() < deadline:
        grad = _log_energy_grad(points)
        # Project gradient onto tangent space
        points_norm = points.copy()
        grad_tangent = grad - np.sum(grad * points_norm, axis=1, keepdims=True) * points_norm
        
        # Adaptive step size
        step = lr * grad_tangent
        new_points = _project_to_sphere(points + step)
        
        new_energy = _log_energy(new_points)
        
        if new_energy < best_energy:
            best_points = new_points.copy()
            best_energy = new_energy
            patience = 0
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.1)
        else:
            patience += 1
            lr = max(lr * 0.9, min_lr)
        
        points = new_points
        
        if patience > max_patience:
            break
    
    # Final polish with smaller steps
    lr = 1e-6
    for _ in range(500):
        if time.time() > deadline:
            break
        grad = _log_energy_grad(points)
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        points = _project_to_sphere(points + lr * grad_tangent)
        e = _log_energy(points)
        if e < best_energy:
            best_points = points.copy()
            best_energy = e
    
    return best_points
# EVOLVE-BLOCK-END
