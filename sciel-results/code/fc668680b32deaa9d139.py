"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _energy_and_grad_var(pts):
    """
    Compute the energy based on the variance of pairwise angular distances
    (via Gram matrix) and its gradient.
    Objective: Minimize sum_{i != j} (p_i . p_j)^2
    This encourages points to be distributed such that their pairwise dot products
    are close to zero, promoting uniformity.
    """
    n = len(pts)
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_n = pts / norms
    
    # Compute pairwise dot products
    dots = pts_n @ pts_n.T
    np.fill_diagonal(dots, 0.0) # Exclude self-interaction
    
    # Energy: Sum of squares of off-diagonal dots
    energy = np.sum(dots * dots)
    
    # Gradient:
    # E = sum_{i != j} (p_i . p_j)^2
    # dE/dp_k = 2 * sum_{j != k} (p_k . p_j) * p_j
    grad = np.zeros_like(pts_n)
    for i in range(n):
        # grad[i] = 2 * sum_{j != i} dots[i, j] * pts_n[j]
        # We can compute this as 2 * (dots @ pts_n) - 2 * dots[i,i] * pts_n[i]
        # But dots[i,i] is 0 in our matrix.
        # So grad[i] = 2 * (dots @ pts_n)[i]
        pass
    
    # Vectorized gradient calculation
    # grad = 2 * (dots @ pts_n)
    grad = 2.0 * (dots @ pts_n)
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . p_i) * p_i
    for i in range(n):
        dot_val = np.dot(grad[i], pts_n[i])
        grad[i] = grad[i] - dot_val * pts_n[i]
        
    return energy, grad


def _covering_radius_fast(pts, n_test=100000):
    """
    Fast estimate of covering radius using a grid of test points.
    """
    M = n_test
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute max dot product for each test point
    dots = test_pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    return np.max(angles)


def _optimize_single(pts_init, max_time=10.0):
    """
    Optimize a single configuration using L-BFGS-B.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pts_reshaped = x.reshape(N, 3)
        e, g = _energy_and_grad_var(pts_reshaped)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    elapsed = time.time() - start_time
    if elapsed > max_time:
        return pts_init
    
    x_opt = result.x
    pts_opt = x_opt.reshape(N, 3)
    
    # Normalize
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_opt = pts_opt / norms
    
    return pts_opt


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses variance minimization of pairwise dot products.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Standard Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with different offset
    initial_configs.append(_fibonacci_spiral(N, offset=0.3))
    
    # 3. Random configurations
    rng = np.random.RandomState(42)
    for _ in range(5):
        random_pts = rng.randn(N, 3)
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / norms
        initial_configs.append(random_pts)
        
    # Optimize each configuration and keep the best
    best_pts = initial_configs[0]
    best_cr = _covering_radius_fast(best_pts)
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Optimize this configuration
        optimized = _optimize_single(config, max_time=min(10.0, time_budget - (time.time() - start_time) - 5))
        
        # Evaluate
        cr = _covering_radius_fast(optimized)
        if cr < best_cr:
            best_cr = cr
            best_pts = optimized
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
