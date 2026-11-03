"""
Minimizing logarithmic energy of 100 points on the unit sphere.

Approach:
1. Start with a good initial configuration (Fibonacci spiral / icosahedral symmetry).
2. Use gradient descent with multiple restarts and annealing to minimize the energy.
3. The energy is sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|).
4. Gradient of energy w.r.t. point i: for each j != i, grad_i += -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|
   Wait, let's be careful.
   E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
   d_ij = |x_i - x_j|
   dE/dx_i = -sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j != i} (x_i - x_j)/d_ij^2
   
   Actually: d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
   
   So grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
   
   We want to minimize E, so we move in direction -grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
   
   Then project back to sphere: x_i = x_i / |x_i|

5. Use adaptive step size and restarts.
"""

import numpy as np
from math import fsum, log, sqrt

N = 100


def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _compute_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total += log(1.0 / d)
    return total


def _compute_energy_fast(points):
    """Faster energy computation using vectorization."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    return np.sum(np.log(1.0 / dist[iu]))


def _compute_gradient(points):
    """Compute gradient of energy w.r.t. each point on the sphere.
    
    E = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / d_ij^2
    
    We return the negative gradient (direction of steepest descent).
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-15:
                continue
            # dE/dx_i contribution from pair (i,j): -diff/d2
            # So -dE/dx_i = diff/d2
            grad[i] += diff / d2
    return grad


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _gradient_descent(points, max_iters=2000, lr=0.01, lr_decay=0.999):
    """Run gradient descent on the sphere."""
    points = points.copy().astype(float)
    current_energy = _compute_energy_fast(points)
    
    for it in range(max_iters):
        grad = _compute_gradient(points)
        # Update: move in direction of negative gradient
        points = points + lr * grad
        points = _project_to_sphere(points)
        
        new_energy = _compute_energy_fast(points)
        
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Reduce learning rate if energy increases
            lr *= 0.5
            if lr < 1e-10:
                break
            # Try again with smaller step
            points = _project_to_sphere(points - lr * grad)
            new_energy = _compute_energy_fast(points)
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                break
        
        lr *= lr_decay
    
    return points, current_energy


def _random_initialization(n, rng):
    """Generate random points on sphere."""
    vecs = rng.standard_normal((n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    best_points = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        if restart == 0:
            # Fibonacci spiral
            points = _fibonacci_spiral(N)
        elif restart == 1:
            # Slightly perturbed Fibonacci
            points = _fibonacci_spiral(N)
            points += rng.standard_normal((N, 3)) * 0.05
            points = _project_to_sphere(points)
        elif restart == 2:
            # Random initialization
            points = _random_initialization(N, rng)
        elif restart == 3:
            # Another random
            points = _random_initialization(N, rng)
        else:
            # Perturbed random
            points = _random_initialization(N, rng)
            points += rng.standard_normal((N, 3)) * 0.1
            points = _project_to_sphere(points)
        
        # Run gradient descent with different learning rates
        for lr in [0.01, 0.005, 0.002]:
            pts, energy = _gradient_descent(points, max_iters=3000, lr=lr, lr_decay=0.9995)
            
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
    
    # Final refinement with very small steps
    if best_points is not None:
        for lr in [0.001, 0.0005, 0.0001]:
            pts, energy = _gradient_descent(best_points, max_iters=2000, lr=lr, lr_decay=0.9999)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
    
    # Ensure all points are on the unit sphere and non-degenerate
    best_points = _project_to_sphere(best_points)
    
    # Check for any points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[i] += rng.standard_normal(3) * 1e-6
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    return best_points
