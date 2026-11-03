"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # seconds, safety margin before 120s limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    # points: (N, 3) array
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Exclude self-pairs (diagonal)
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    # Avoid log(0) or log of very small numbers
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_and_grad(points):
    """Compute energy and gradient. Returns (energy, gradient)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    dists_sq_safe = np.maximum(dists_sq, 1e-30)
    
    # Gradient of log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|
    # = -(x_i - x_j) / |x_i - x_j|^2
    # For point i: sum over j != i of (x_j - x_i) / |x_i - x_j|^2
    
    # Build gradient
    grad = np.zeros_like(points)
    # grad[i] = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Vectorized: for each pair (i,j), contribution to grad[i] is (x_j - x_i)/d^2
    # and to grad[j] is (x_i - x_j)/d^2 = -(x_j - x_i)/d^2
    
    # Use the full matrix approach
    # inv_d2[i,j] = 1/|x_i - x_j|^2 for i != j, 0 for i == j
    inv_d2 = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-30
    inv_d2[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_d2[i,j]
    # = (sum_j x_j * inv_d2[i,j]) - x_i * sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)
    weighted_sum = inv_d2 @ points  # shape (N, 3)
    grad = weighted_sum - points * row_sums[:, np.newaxis]
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    e = math.fsum(np.log(1.0 / d).tolist())
    
    return e, grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(points * points, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent_sphere(points, max_iters=5000, lr_init=0.01, tol=1e-12):
    """
    Gradient descent on the sphere using retraction.
    At each step: compute gradient, move in negative gradient direction, project back to sphere.
    """
    points = _project_to_sphere(points.copy())
    best_energy = _energy(points)
    best_points = points.copy()
    
    lr = lr_init
    for it in range(max_iters):
        if _elapsed() > DEADLINE:
            break
        
        e, grad = _energy_and_grad(points)
        
        if e < best_energy:
            best_energy = e
            best_points = points.copy()
        
        # Check convergence
        grad_norm = np.sqrt(np.sum(grad * grad))
        if grad_norm < tol:
            break
        
        # Adaptive step size
        # Move: p_new = p - lr * grad
        p_new = points - lr * grad
        
        # Project to sphere
        p_new = _project_to_sphere(p_new)
        
        # Check if energy improved
        e_new = _energy(p_new)
        if e_new < e:
            points = p_new
            # Decay learning rate slightly
            lr *= 0.999
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
        
        if it % 100 == 0 and _elapsed() > DEADLINE:
            break
    
    return best_points, best_energy


def _simulated_annealing(points, max_iters=30000, T_init=0.1, T_min=1e-6, alpha=0.9995):
    """
    Simulated annealing on the sphere.
    Propose random perturbations, accept based on Metropolis criterion.
    """
    points = _project_to_sphere(points.copy())
    current_energy = _energy(points)
    best_energy = current_energy
    best_points = points.copy()
    
    T = T_init
    n_accepted = 0
    n_total = 0
    
    for it in range(max_iters):
        if _elapsed() > DEADLINE:
            break
        
        # Propose a random perturbation
        # Perturb a random subset of points with random directions
        n_perturb = np.random.randint(1, 6)  # 1-5 points
        indices = np.random.choice(N, size=n_perturb, replace=False)
        perturbations = np.random.randn(n_perturb, 3) * 0.1
        
        p_new = points.copy()
        p_new[indices] += perturbations
        p_new = _project_to_sphere(p_new)
        
        # Check for too-close points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p_new[i] - p_new[j])
                if d < min_dist:
                    min_dist = d
                if d < 1e-9:
                    break
            if min_dist < 1e-9:
                break
        
        if min_dist < 1e-9:
            continue
        
        e_new = _energy(p_new)
        delta_e = e_new - current_energy
        n_total += 1
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-12)):
            points = p_new
            current_energy = e_new
            n_accepted += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        
        T = max(T * alpha, T_min)
        
        # Periodic gradient polish
        if it % 5000 == 4999 and _elapsed() < DEADLINE - 30:
            points, _ = _gradient_descent_sphere(points, max_iters=500, lr_init=0.005)
            current_energy = _energy(points)
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
    
    return best_points, best_energy


def _local_search(points, max_iters=20000, step_size=0.05):
    """
    Local search with random restarts within neighborhood.
    """
    points = _project_to_sphere(points.copy())
    current_energy = _energy(points)
    best_energy = current_energy
    best_points = points.copy()
    
    for it in range(max_iters):
        if _elapsed() > DEADLINE:
            break
        
        # Propose perturbation
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        perturbations = np.random.randn(n_perturb, 3) * step_size
        
        p_new = points.copy()
        p_new[indices] += perturbations
        p_new = _project_to_sphere(p_new)
        
        e_new = _energy(p_new)
        
        if e_new < current_energy:
            points = p_new
            current_energy = e_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            # Occasionally accept worse solutions to escape local minima
            if np.random.random() < 0.05:
                points = p_new
                current_energy = e_new
    
    return best_points, best_energy


def construct():
    global START_TIME
    START_TIME = time.time()
    
    np.random.seed(42)
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # First, do gradient descent to get to a local minimum
    points, e1 = _gradient_descent_sphere(points, max_iters=3000, lr_init=0.01)
    
    # Now do simulated annealing to explore
    if _elapsed() < DEADLINE - 20:
        points, e2 = _simulated_annealing(points, max_iters=20000, T_init=0.05, T_min=1e-6, alpha=0.9998)
    
    # Final gradient polish
    if _elapsed() < DEADLINE - 5:
        points, e3 = _gradient_descent_sphere(points, max_iters=10000, lr_init=0.005, tol=1e-14)
    
    # Ensure valid output
    points = _project_to_sphere(points)
    
    # Check for too-close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                # Nudge point j slightly
                points[j] += np.random.randn(3) * 1e-6
                points = _project_to_sphere(points)
                break
    
    # Final check
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                # Force separation
                points[j] = points[i] + np.random.randn(3) * 0.01
                points = _project_to_sphere(points)
    
    # Verify all finite
    if not np.all(np.isfinite(points)):
        points = _fibonacci_sphere(N)
        points, _ = _gradient_descent_sphere(points, max_iters=1000, lr_init=0.01)
        points = _project_to_sphere(points)
    
    return points
