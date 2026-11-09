"""Pack 50 equal circles in a unit circle; maximize common radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _initial_hex_grid(n):
    """Generate a hexagonal lattice of points, then trim to fit in unit disk with margin."""
    # Start with a spacing that gives roughly n points in the disk
    # Area of disk = pi, area per hex cell ~ (sqrt(3)/2) * d^2 where d is spacing
    # For n points: n * (sqrt(3)/2) * d^2 ~ pi  =>  d ~ sqrt(2*pi/(n*sqrt(3)))
    d = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex grid points
    points = []
    max_r = 1.5
    k_max = int(np.ceil(max_r / d)) + 1
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * d
            y = j * d * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - d / 2.0 - 0.01:  # small margin
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n, pick the n closest to center
    if len(points) > n:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n]
        points = points[idx]
    elif len(points) < n:
        # Fallback: reduce spacing and retry
        while len(points) < n and d > 0.01:
            d *= 0.95
            points = []
            k_max = int(np.ceil(1.5 / d)) + 1
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * d
                    y = j * d * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - d / 2.0 - 0.01:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > n:
                radii = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(radii, kind='stable')[:n]
                points = points[idx]
    
    return points[:n]


def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    n = len(centers)
    # Min distance from center to boundary
    boundary_min = 1.0 - np.max(np.hypot(centers[:, 0], centers[:, 1]))
    
    # Min pairwise distance / 2
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(boundary_min, min_pair_dist / 2.0)


def _objective(x, n=N):
    """Objective function: negative of the radius (we maximize radius, so minimize -radius).
    Uses a smooth approximation to the min via log-sum-exp."""
    centers = x.reshape(n, 2)
    
    # Boundary constraint: 1 - |c_i|
    radii = np.hypot(centers[:, 0], centers[:, 1])
    boundary_vals = 1.0 - radii
    
    # Pairwise distances / 2
    pair_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            pair_vals.append(d / 2.0)
    pair_vals = np.array(pair_vals)
    
    # Soft min using log-sum-exp: -log(sum(exp(-beta * vals))) / beta
    beta = 50.0
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Shift for numerical stability
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * min_val
    soft_min = -log_sum_exp / beta
    
    return -soft_min


def _gradient(x, n=N):
    """Gradient of the objective function."""
    centers = x.reshape(n, 2)
    beta = 50.0
    
    # Compute all values
    boundary_vals = np.zeros(n)
    for i in range(n):
        boundary_vals[i] = 1.0 - np.hypot(centers[i, 0], centers[i, 1])
    
    pair_vals = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            pair_vals[i, j] = pair_vals[j, i] = d / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals.flatten()])
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft min = weighted average of gradients of each term
    grad = np.zeros((n, 2))
    
    # Boundary terms
    for i in range(n):
        r = np.hypot(centers[i, 0], centers[i, 1])
        if r > 1e-10:
            grad[i, 0] += weights[i] * (-centers[i, 0] / r)
            grad[i, 1] += weights[i] * (-centers[i, 1] / r)
    
    # Pairwise terms
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[n + idx]
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if d > 1e-10:
                grad[i, 0] += w * (dx / d)
                grad[i, 1] += w * (dy / d)
                grad[j, 0] -= w * (dx / d)
                grad[j, 1] -= w * (dy / d)
            idx += 1
    
    # Objective is -soft_min, so gradient is -grad
    return -grad.reshape(-1)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Initial placement: hexagonal grid
    centers = _initial_hex_grid(N)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    x0 = centers.flatten()
    
    # Run L-BFGS-B optimization
    max_iter = 2000
    try:
        result = minimize(
            _objective,
            x0,
            jac=_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if result.x is not None:
            new_centers = result.x.reshape(N, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_centers
    
    # Second pass: refine with tighter settings
    try:
        x0 = best_centers.flatten()
        result2 = minimize(
            _objective,
            x0,
            jac=_gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        
        if result2.x is not None:
            new_centers = result2.x.reshape(N, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_centers
    
    # Final polish: local search with small perturbations
    try:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            # Try small random perturbations
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                perturbed = best_centers + np.random.normal(0, 0.001, (N, 2))
                # Project back: ensure inside unit circle
                radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    perturbed[mask] *= (1.0 / radii[mask])[:, None]
                
                r = _compute_radius(perturbed)
                if r > best_radius:
                    best_centers = perturbed
                    best_radius = r
    except Exception:
        pass
    
    return best_centers
# EVOLVE-BLOCK-END
