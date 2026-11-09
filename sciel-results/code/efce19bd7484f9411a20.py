"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _fibonacci_lattice_init(n, seed=42):
    """Generate points on a Fibonacci lattice scaled to [-0.5, 0.5]^3."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    points = []
    for i in range(n):
        # Fibonacci sphere points
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / phi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Scale to fit in [-0.5, 0.5]^3
    points = points * 0.45  # leave margin from walls
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.01, size=points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.45, 0.45)
    
    return points


def _min_radius(x):
    """Compute the maximum radius for a configuration."""
    x = x.reshape(N, 3)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    # Distance to walls
    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    min_wall_dist = np.min(wall_dists)

    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_radius(x, beta=50.0):
    """Smooth approximation of the minimum radius using log-sum-exp."""
    x = x.reshape(N, 3)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distance to walls
    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    
    # Combine all distances (pairwise/2 and wall distances)
    all_dists = np.concatenate([dists[dists < np.inf] / 2.0, wall_dists])
    
    # Soft minimum using log-sum-exp
    min_val = np.min(all_dists)
    soft_min = min_val - np.log(np.sum(np.exp(-beta * (all_dists - min_val)))) / beta
    
    return soft_min


def _neg_soft_min_radius(x, beta=50.0):
    """Negative soft minimum for minimization."""
    return -_soft_min_radius(x, beta)


def _gradient_soft_min_radius(x, beta=50.0):
    """Numerical gradient of the negative soft minimum radius."""
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_neg_soft_min_radius(x_plus, beta) - _neg_soft_min_radius(x_minus, beta)) / (2 * eps)
    return grad


def _local_refinement(x, beta_start=50.0, beta_end=500.0, n_iters=30):
    """Refine configuration using gradient-based optimization with increasing beta."""
    x = x.copy()
    for i in range(n_iters):
        if time.time() > _deadline:
            break
        # Interpolate beta
        beta = beta_start + (beta_end - beta_start) * (i / max(n_iters - 1, 1))
        
        # Compute numerical gradient
        grad = _gradient_soft_min_radius(x, beta)
        
        # Simple gradient descent with adaptive step size
        lr = 0.01 / (1 + i * 0.1)
        x_new = x - lr * grad
        x_new = np.clip(x_new, -0.5, 0.5)
        
        # Accept if it improves
        if _min_radius(x_new) >= _min_radius(x):
            x = x_new
    
    return x


def construct():
    global _deadline
    t0 = time.time()
    _deadline = t0 + 130.0  # safety margin before 150s kill
    
    # Initialize with Fibonacci lattice
    x0 = _fibonacci_lattice_init(N)
    
    best_x = x0.copy()
    best_score = _min_radius(best_x)
    
    # Try multiple restarts with different seeds
    for restart in range(8):
        if time.time() > _deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Different initialization seeds
            x_init = _fibonacci_lattice_init(N, seed=restart * 100 + 7)
        
        # Local refinement
        x_refined = _local_refinement(x_init)
        
        score = _min_radius(x_refined)
        if score > best_score:
            best_score = score
            best_x = x_refined.copy()
    
    # Final polish with scipy minimize using the non-smooth objective
    if time.time() < _deadline - 20:
        def neg_score(x):
            return -_min_radius(x)
        
        result = minimize(
            neg_score,
            best_x.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_result = result.x.reshape(N, 3)
        if _min_radius(x_result) > best_score:
            best_score = _min_radius(x_result)
            best_x = x_result
    
    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)
    
    # Ensure no coincident centers
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-6 * np.array([1, 0, 0])
    
    return best_x
# EVOLVE-BLOCK-END
