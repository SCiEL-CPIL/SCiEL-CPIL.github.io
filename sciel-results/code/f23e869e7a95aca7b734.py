"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _covering_radius(points):
    """Compute covering radius via convex hull facets (exact)."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_angle = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        facet_radius = max(d1, d2, d3)
        max_angle = max(max_angle, facet_radius)
    return max_angle

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _create_grid(n_theta=200, n_phi=400):
    """Create a dense spherical grid for coverage field evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _coverage_field(points, grid, sigma=0.3):
    """Compute smooth coverage field using Gaussian RBF kernel.
    Coverage at grid point g = sum_i exp(-||g - p_i||^2 / (2*sigma^2))
    Higher coverage = closer to a point. We want to MINIMIZE the maximum
    distance, which means MAXIMIZE the minimum coverage.
    Equivalent: minimize -min_coverage, or use negative log-sum-exp for smoothness.
    """
    # points: (N, 3), grid: (M, 3)
    # Compute squared distances: ||g - p||^2 = ||g||^2 + ||p||^2 - 2*g.p
    # Since all on unit sphere: ||g - p||^2 = 2 - 2*g.p
    dots = grid @ points.T  # (M, N)
    sq_dists = 2.0 - 2.0 * dots  # (M, N)
    # Clip for numerical stability
    sq_dists = np.maximum(sq_dists, 0.0)
    # Gaussian kernel
    kernel = np.exp(-sq_dists / (2.0 * sigma**2))  # (M, N)
    # Coverage at each grid point
    coverage = kernel.sum(axis=1)  # (M,)
    return coverage

def _objective_and_grad(flat_points, grid, sigma=0.3):
    """Objective: negative minimum coverage (smooth min-max).
    We minimize -min(coverage) which is equivalent to maximizing min coverage.
    For smoothness, use soft-min: -logsumexp(-coverage) / M as a smooth approximation.
    Actually, let's use: maximize min coverage -> minimize -min coverage
    Smooth version: -log(sum(exp(-coverage_i / tau))) * tau / M
    But simpler: just minimize the negative of the minimum, with subgradient.
    Let's use a differentiable approach: minimize the p-norm of (max_coverage - coverage)
    Or better: minimize -logsumexp(coverage / tau) which approximates -max(coverage)
    
    Actually, we want to maximize the minimum coverage.
    Let's define: f = min_i coverage_i
    Smooth approximation: f_smooth = -tau * log(sum_i exp(-coverage_i / tau))
    We want to maximize f_smooth, so minimize -f_smooth = tau * log(sum_i exp(-coverage_i / tau))
    """
    points = flat_points.reshape(-1, 3)
    # Re-normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    coverage = _coverage_field(points, grid, sigma)
    
    # Smooth min: -tau * log(sum(exp(-coverage/tau)))
    tau = 0.05
    # For numerical stability
    c_min = np.min(coverage)
    exp_terms = np.exp(-(coverage - c_min) / tau)
    smooth_min = c_min - tau * np.log(np.sum(exp_terms))
    
    # We want to maximize smooth_min, so objective = -smooth_min
    obj = -smooth_min
    
    # Gradient with respect to flat_points
    # d(smooth_min)/d(coverage_i) = -exp_terms[i] / sum(exp_terms)
    weights = exp_terms / np.sum(exp_terms)  # (M,)
    
    # d(coverage_i)/d(points_j) for each grid point i and point j
    # coverage_i = sum_j exp(-sq_dist_ij / (2*sigma^2))
    # d(coverage_i)/d(points_j) = exp(-sq_dist_ij / (2*sigma^2)) * (grid_i - points_j) / sigma^2
    # But we need to account for the re-normalization constraint.
    # For simplicity, compute gradient on the sphere via projection.
    
    # Compute gradient of -smooth_min w.r.t. points
    # d(obj)/d(points_j) = -sum_i weights[i] * d(coverage_i)/d(points_j)
    # d(coverage_i)/d(points_j) = kernel_ij * (grid_i - points_j) / sigma^2
    
    dots = grid @ points.T  # (M, N)
    sq_dists = np.maximum(2.0 - 2.0 * dots, 0.0)
    kernel = np.exp(-sq_dists / (2.0 * sigma**2))  # (M, N)
    
    # grad_points[j] = -sum_i weights[i] * kernel[i,j] * (grid[i] - points[j]) / sigma^2
    # = -1/sigma^2 * sum_i (weights[i] * kernel[i,j]) * (grid[i] - points[j])
    # = -1/sigma^2 * [ (weights * kernel) @ grid - (weights * kernel).sum(axis=0, keepdims=True) * points ]
    
    weighted_kernel = weights[:, None] * kernel  # (M, N)
    sum_wk = weighted_kernel.sum(axis=0, keepdims=True)  # (1, N)
    wk_grid = weighted_kernel.T @ grid  # (N, 3)
    
    grad_points = -(wk_grid - sum_wk * points) / (sigma**2)  # (N, 3)
    
    # Project gradient onto tangent space (remove radial component)
    # grad_tangent = grad - (grad . p) * p
    radial = (grad_points * points).sum(axis=1, keepdims=True) * points
    grad_tangent = grad_points - radial
    
    return obj, grad_tangent.ravel()

def _lbfgs_optimize(points, grid, sigma=0.3, maxiter=100, deadline=None):
    """Optimize using L-BFGS-B on the smooth coverage objective."""
    flat = points.ravel().copy()
    
    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            # Return large value to stop
            return 1e10, np.zeros_like(x)
        obj, grad = _objective_and_grad(x, grid, sigma)
        return obj, grad
    
    result = minimize(fun_and_grad, flat, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': 1e-6, 'maxfun': 200})
    
    opt_points = result.x.reshape(-1, 3)
    # Re-normalize
    norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    opt_points = opt_points / norms
    
    return opt_points

def _repulsion_optimize(points, max_iters=300, lr=0.01, deadline=None):
    """Quick repulsion-based optimization for initial spreading."""
    points = points.copy().astype(float)
    n = len(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        dots = np.clip(points @ points.T, -1, 1)
        angles = np.arccos(dots)
        np.fill_diagonal(angles, 0)
        
        threshold = np.pi / 3
        mask = angles < threshold
        np.fill_diagonal(mask, False)
        
        weights = np.where(mask, 1.0 / (1.0 + angles**2), 0.0)
        weight_sums = weights.sum(axis=1, keepdims=True)
        forces = weight_sums * points - (weights @ points)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms > 1e-10, force_norms, 1.0)
        forces = forces / force_norms
        
        radial = (forces * points).sum(axis=1, keepdims=True) * points
        tangent_forces = forces - radial
        
        points = points + lr * tangent_forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Create grid for smooth optimization
    grid = _create_grid(n_theta=150, n_phi=300)
    
    # Phase 1: Fibonacci initialization + quick repulsion
    points = _fibonacci_sphere(N)
    if time.time() < deadline - 40:
        points = _repulsion_optimize(points, max_iters=100, lr=0.005, deadline=deadline - 35)
    
    radius = _covering_radius(points)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Phase 2: L-BFGS-B smooth optimization from multiple starting points
    sigma_values = [0.3, 0.25, 0.2]
    
    for sigma in sigma_values:
        if time.time() > deadline - 20:
            break
        
        # Try from current best
        if best_points is not None:
            opt_points = _lbfgs_optimize(best_points, grid, sigma=sigma, 
                                          maxiter=80, deadline=deadline - 15)
            opt_radius = _covering_radius(opt_points)
            if opt_radius < best_radius:
                best_radius = opt_radius
                best_points = opt_points.copy()
        
        # Try from different Fibonacci offset
        if time.time() < deadline - 25:
            points2 = _fibonacci_sphere(N, offset=0.3)
            points2 = _repulsion_optimize(points2, max_iters=50, lr=0.005, 
                                           deadline=deadline - 20)
            opt_points2 = _lbfgs_optimize(points2, grid, sigma=sigma,
                                           maxiter=60, deadline=deadline - 10)
            opt_radius2 = _covering_radius(opt_points2)
            if opt_radius2 < best_radius:
                best_radius = opt_radius2
                best_points = opt_points2.copy()
    
    # Phase 3: Final polish with smaller sigma
    if best_points is not None and time.time() < deadline - 5:
        final = _lbfgs_optimize(best_points, grid, sigma=0.15,
                                 maxiter=50, deadline=deadline - 2)
        final_radius = _covering_radius(final)
        if final_radius < best_radius:
            best_points = final
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
