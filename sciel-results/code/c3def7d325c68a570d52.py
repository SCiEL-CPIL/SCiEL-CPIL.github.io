"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _energy_and_grad(pts):
    """
    Compute the electrostatic energy and its gradient for points on the sphere.
    Energy: sum_{i != j} 1 / (1 - dot(pi, pj))
    This encourages points to spread out uniformly.
    """
    n = len(pts)
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_n = pts / norms
    
    # Compute pairwise dot products
    dots = pts_n @ pts_n.T
    np.fill_diagonal(dots, 1.0)
    
    # Avoid numerical issues
    dots = np.clip(dots, -1.0, 0.999999)
    
    # Energy: sum of 1/(1-dots) for i != j
    inv_dists = 1.0 / (1.0 - dots)
    np.fill_diagonal(inv_dists, 0.0)
    energy = np.sum(inv_dists)
    
    # Gradient computation
    # d/dpi [1/(1 - pi·pj)] = pj / (1 - pi·pj)^2 for i != j
    # The gradient with respect to pi is sum_{j != i} pj / (1 - pi·pj)^2
    # But we need to account for the constraint that points are on the sphere.
    # We compute the gradient in R^3 and then project to the tangent space.
    
    grad = np.zeros_like(pts_n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = 1.0 - dots[i, j]
            grad[i] += pts_n[j] / (d * d)
    
    # Project gradient to tangent space: grad_tangent = grad - (grad·pi) * pi
    for i in range(n):
        dot_val = np.dot(grad[i], pts_n[i])
        grad[i] = grad[i] - dot_val * pts_n[i]
    
    return energy, grad


def _covering_radius_fast(pts, n_test=50000):
    """
    Fast estimate of covering radius using a grid of test points.
    """
    # Create a uniform grid on the sphere
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


def _optimize_single(pts_init, max_time=90.0):
    """
    Optimize a single configuration using L-BFGS-B with analytical gradient.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pts_reshaped = x.reshape(N, 3)
        e, g = _energy_and_grad(pts_reshaped)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B with analytical gradient
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
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
    Uses energy minimization with multiple restarts.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Standard Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with different offset
    initial_configs.append(_fibonacci_spiral(N, offset=0.3))
    
    # 3. Random configuration with seeding
    rng = np.random.RandomState(42)
    for _ in range(3):
        # Generate random points
        random_pts = rng.randn(N, 3)
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / norms
        initial_configs.append(random_pts)
    
    # 4. Add some structured perturbations to Fibonacci
    base = _fibonacci_spiral(N)
    for angle in [0.1, 0.2, 0.3]:
        perturbed = base.copy()
        # Apply small random rotations
        theta = angle * rng.randn(3)
        # Simple rotation around z-axis
        phi_rot = angle * rng.randn()
        c, s = np.cos(phi_rot), np.sin(phi_rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        perturbed = perturbed @ R.T
        initial_configs.append(perturbed)
    
    # Optimize each configuration and keep the best
    best_pts = initial_configs[0]
    best_cr = _covering_radius_fast(best_pts)
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Optimize this configuration
        optimized = _optimize_single(config, max_time=min(15.0, time_budget - (time.time() - start_time) - 5))
        
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
