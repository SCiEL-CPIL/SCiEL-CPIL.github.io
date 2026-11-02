"""Rewritten program: differentiable rigid-body mechanics approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # For n circles in unit disk, area per circle ~ pi/n, so radius ~ sqrt(1/n)
    # Hexagonal spacing ~ 2*r
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.05
    
    # Generate hexagonal lattice points
    points = []
    k_max = int(1.0 / spacing) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points that fit (with some margin)
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        # If not enough, reduce spacing
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 3
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    # Sort by distance from center and take first n
    idx = np.argsort(radii)
    return points[idx[:n]]


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Minimum distance between any pair / 2
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Minimum distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_penalty(centers, target_dist, beta=100.0):
    """
    Compute a smooth approximation to the maximum violation using log-sum-exp.
    Returns (penalty, gradient) as a tuple.
    """
    n = len(centers)
    c = centers
    
    # Pairwise violations: target_dist - d_ij for d_ij < target_dist
    # Boundary violations: r_i - (1 - target_dist/2) for r_i > 1 - target_dist/2
    
    violations = []
    grads = []
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            diff = c[i] - c[j]
            d = np.sqrt(np.sum(diff ** 2) + 1e-10)
            if d < target_dist:
                v = target_dist - d
                # Gradient of v with respect to c[i] and c[j]
                # dv/dc[i] = diff / d
                # dv/dc[j] = -diff / d
                gv = np.zeros(2 * n)
                gv[2*i:2*i+2] = diff / d
                gv[2*j:2*j+2] = -diff / d
                violations.append(v)
                grads.append(gv)
    
    # Boundary
    for i in range(n):
        c_i = c[i]
        r_i = np.sqrt(np.sum(c_i ** 2) + 1e-10)
        if r_i > 1.0 - target_dist / 2.0:
            v = r_i - (1.0 - target_dist / 2.0)
            gv = np.zeros(2 * n)
            gv[2*i:2*i+2] = c_i / r_i
            violations.append(v)
            grads.append(gv)
    
    if len(violations) == 0:
        return 0.0, np.zeros(2 * n)
    
    violations = np.array(violations)
    grads = np.array(grads)  # shape (m, 2*n)
    
    # Soft max: (1/beta) * log(sum(exp(beta * v)))
    # Use numerically stable version
    max_v = np.max(violations)
    if max_v < 0:
        max_v = 0.0
    
    exp_v = np.exp(beta * (violations - max_v))
    penalty = (max_v + np.log(np.sum(exp_v)) / beta)
    
    # Gradient: sum(exp(beta*v) * beta * grad_v) / sum(exp(beta*v))
    weights = exp_v / np.sum(exp_v)
    gradient = weights @ grads
    
    return penalty, gradient


def _optimize_packing(centers, target_dist, max_iter=500):
    """
    Optimize the packing using gradient descent to minimize the maximum violation.
    """
    n = len(centers)
    
    def objective(x):
        c = x.reshape(-1, 2)
        penalty, _ = _soft_min_penalty(c, target_dist, beta=100.0)
        return penalty
    
    def gradient(x):
        c = x.reshape(-1, 2)
        _, grad = _soft_min_penalty(c, target_dist, beta=100.0)
        return grad
    
    x0 = centers.flatten()
    
    # Use L-BFGS-B for optimization
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    return result.x.reshape(-1, 2)


def _binary_search_radius(centers, low, high, tol=1e-7, max_iter=25):
    """
    Binary search on the radius to find the maximum feasible packing radius.
    """
    best_centers = centers.copy()
    best_radius = low
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        
        # Try to find a valid configuration for radius mid
        optimized = _optimize_packing(centers, 2.0 * mid, max_iter=300)
        
        # Check if the optimized configuration is valid
        score = _compute_score(optimized)
        
        if score >= mid - tol:
            # Feasible, try larger radius
            low = mid
            best_centers = optimized.copy()
            best_radius = mid
        else:
            # Not feasible, try smaller radius
            high = mid
        
        if high - low < tol:
            break
    
    return best_centers, best_radius


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses a differentiable rigid-body mechanics approach with soft-min penalty
    and binary search on the radius.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Estimate initial radius
    initial_score = _compute_score(centers)
    
    # Binary search on the radius
    low = initial_score * 0.85
    high = initial_score * 1.15
    
    # Try to improve with binary search
    optimized, final_radius = _binary_search_radius(centers, low, high, tol=1e-7, max_iter=25)
    
    # Verify the result
    final_score = _compute_score(optimized)
    
    # If the score is worse than the initial, fall back to the initial
    if final_score < initial_score:
        optimized = centers
        final_score = initial_score
    
    # Ensure all points are within the unit circle
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0 - final_score:
            # Scale back to just inside the boundary
            scale = (1.0 - final_score - 1e-10) / r
            optimized[i] *= scale
    
    return optimized
# EVOLVE-BLOCK-END
