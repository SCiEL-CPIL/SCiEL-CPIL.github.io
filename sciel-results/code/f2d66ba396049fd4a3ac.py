"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for each point x_i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    # Energy: -sum_{i<j} log(|x_i - x_j|)
    # = -0.5 * sum_{i != j} log(|x_i - x_j|)
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: diff[i,j] = x_i - x_j, so we need (x_i - x_j) / |x_i - x_j|^2
    # = diff / dists_sq
    grad = -np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_riemannian(points, deadline, maxiter=500):
    """
    High-precision Riemannian optimization using L-BFGS-B.
    """
    if time.time() > deadline - 10:
        return points

    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        energy, grad = _log_energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_tangent = np.zeros_like(grad)
        for i in range(len(pts)):
            proj = np.dot(grad[i], pts[i])
            grad_tangent[i] = grad[i] - proj * pts[i]
        
        return energy, grad_tangent.flatten()

    try:
        result = minimize(
            objective_and_grad,
            points.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxls': 50}
        )
        return _normalize(result.x.reshape(-1, 3))
    except Exception:
        return points


def _simulated_annealing(points, deadline, rng):
    """
    Simulated Annealing on the sphere.
    """
    current_points = points.copy()
    current_energy, _ = _log_energy_and_grad(current_points)
    
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    start_time = time.time()
    
    while T > T_min and time.time() < deadline - 20:
        # Move 1-3 random points
        n_move = rng.integers(1, 4)
        idx = rng.choice(N, size=n_move, replace=False)
        
        # Generate random tangent vectors
        new_points = current_points.copy()
        for i in idx:
            # Random vector
            v = rng.standard_normal(3)
            # Project to tangent space
            v_tan = v - np.dot(v, current_points[i]) * current_points[i]
            norm_v = np.linalg.norm(v_tan)
            if norm_v > 1e-12:
                v_tan = v_tan / norm_v
            else:
                v_tan = rng.standard_normal(3)
                v_tan = v_tan - np.dot(v_tan, current_points[i]) * current_points[i]
                v_tan = v_tan / np.linalg.norm(v_tan)
            
            # Step size proportional to sqrt(T)
            step_size = 0.1 * np.sqrt(T)
            new_points[i] = current_points[i] + step_size * v_tan
            new_points[i] = _normalize(new_points[i][:np.newaxis, :])[0]
            
        new_energy, _ = _log_energy_and_grad(new_points)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= alpha
        
        # Periodic polish to escape local minima more effectively
        if time.time() > start_time + 5:
            start_time = time.time()
            # Small polish
            polished = _optimize_riemannian(current_points, deadline, maxiter=20)
            e_polished, _ = _log_energy_and_grad(polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_points = polished.copy()
            current_points = polished
            current_energy = e_polished

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Initial Fibonacci refinement
    try:
        if time.time() < deadline - 30:
            refined = _optimize_riemannian(best_points, deadline, maxiter=100)
            e, _ = _log_energy_and_grad(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    except Exception:
        pass

    rng = np.random.default_rng(42)
    
    # 2. Simulated Annealing with multiple restarts
    num_restarts = 3
    for k in range(num_restarts):
        if time.time() > deadline - 40:
            break
            
        # Start from a random or perturbed configuration
        if k == 0:
            start_pts = _fibonacci_sphere(N, seed=1.0)
        elif k == 1:
            start_pts = _random_sphere(N, rng)
        else:
            # Perturb the current best
            start_pts = best_points.copy()
            idx = rng.choice(N, size=20, replace=False)
            for i in idx:
                v = rng.standard_normal(3)
                v_tan = v - np.dot(v, start_pts[i]) * start_pts[i]
                start_pts[i] = start_pts[i] + 0.1 * v_tan / np.linalg.norm(v_tan)
                start_pts[i] = _normalize(start_pts[i][:np.newaxis, :])[0]
            
        # Run SA
        sa_result = _simulated_annealing(start_pts, deadline, rng)
        e_sa, _ = _log_energy_and_grad(sa_result)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_points = sa_result

    # 3. Final high-precision polish
    if time.time() < deadline - 5:
        final_pts = _optimize_riemannian(best_points, deadline, maxiter=500)
        e_final, _ = _log_energy_and_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_pts

    return _normalize(best_points)

# EVOLVE-BLOCK-END
