"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _gram_variance_and_grad(pts):
    """
    Compute variance of off-diagonal Gram matrix entries and its gradient.
    For uniform distribution on S^2, off-diagonal entries should be ~0.
    Minimizing their variance promotes uniform angular spacing.
    """
    n = len(pts)
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_n = pts / norms
    
    # Compute Gram matrix
    G = pts_n @ pts_n.T
    np.fill_diagonal(G, 0.0)
    
    # Off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off_diag = G[mask]
    
    # Variance of off-diagonal entries
    mean_val = np.mean(off_diag)
    var_val = np.mean((off_diag - mean_val) ** 2)
    
    # Also add a term to push mean toward 0 (for true uniformity)
    energy = var_val + 0.1 * mean_val ** 2
    
    # Gradient: d/dP_ij [variance of off-diag(G)]
    # G = P @ P.T, so dG/dP_kl = delta_ki * P_jl + delta_lj * P_ik
    # For variance: d/dG_ij (off-diag) = 2*(G_ij - mean) / (n*(n-1))
    # Then chain rule through G = P @ P.T
    
    # Gradient of variance w.r.t. G (off-diagonal only)
    dG_dvar = np.zeros((n, n))
    dG_dvar[mask] = 2.0 * (G[mask] - mean_val) / (n * (n - 1))
    dG_dvar += 0.1 * 2.0 * mean_val / (n * (n - 1)) * mask  # gradient of mean^2 term
    
    # Gradient w.r.t. P: dE/dP_i = sum_j dE/dG_ij * P_j + sum_j dE/dG_ji * P_j
    # Since G is symmetric, dE/dG_ij = dE/dG_ji
    grad = dG_dvar @ pts_n + dG_dvar.T @ pts_n
    
    # Project to tangent space
    for i in range(n):
        dot_val = np.dot(grad[i], pts_n[i])
        grad[i] = grad[i] - dot_val * pts_n[i]
    
    return energy, grad


def _covering_radius_fast(pts, n_test=80000):
    """
    Fast estimate of covering radius using a grid of test points.
    Uses a denser grid for better accuracy.
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


def _optimize_single(pts_init, max_time=60.0):
    """
    Optimize a single configuration using L-BFGS-B with Gram variance objective.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pts_reshaped = x.reshape(N, 3)
        e, g = _gram_variance_and_grad(pts_reshaped)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B with analytical gradient
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
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


def _random_points(n, rng):
    """Generate random points on the sphere."""
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _rotate_points(pts, rng, max_angle=0.5):
    """Apply a random rotation to points."""
    # Random rotation matrix
    u = rng.randn(3)
    u /= np.linalg.norm(u)
    angle = rng.uniform(0, max_angle)
    c, s = np.cos(angle), np.sin(angle)
    # Rodrigues' rotation formula
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return pts @ R.T


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses Gram matrix variance minimization with multiple restarts.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    rng = np.random.RandomState(12345)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral variants
    initial_configs.append(_fibonacci_spiral(N, offset=0.5))
    initial_configs.append(_fibonacci_spiral(N, offset=0.25))
    initial_configs.append(_fibonacci_spiral(N, offset=0.75))
    
    # 2. Random configurations
    for _ in range(4):
        initial_configs.append(_random_points(N, rng))
    
    # 3. Perturbed Fibonacci
    base = _fibonacci_spiral(N, offset=0.5)
    for _ in range(3):
        initial_configs.append(_rotate_points(base, rng, max_angle=0.3))
    
    # Optimize each configuration and keep the best
    best_pts = initial_configs[0]
    best_cr = _covering_radius_fast(best_pts)
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > time_budget - 10:
            break
        
        remaining = time_budget - (time.time() - start_time) - 10
        if remaining < 5:
            break
        
        # Optimize this configuration
        optimized = _optimize_single(config, max_time=min(remaining, 30.0))
        
        # Evaluate
        cr = _covering_radius_fast(optimized)
        if cr < best_cr:
            best_cr = cr
            best_pts = optimized
    
    # Final refinement: do one more optimization pass on the best solution
    if time.time() - start_time < time_budget - 15:
        final_opt = _optimize_single(best_pts, max_time=min(15.0, time_budget - (time.time() - start_time) - 10))
        final_cr = _covering_radius_fast(final_opt)
        if final_cr < best_cr:
            best_pts = final_opt
            best_cr = final_cr
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
