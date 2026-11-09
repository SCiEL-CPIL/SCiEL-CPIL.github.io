"""Pack 50 equal circles in a unit circle; maximize common radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _initial_hex_grid(n):
    """Generate a hexagonal lattice of points, then trim to fit in unit disk with margin."""
    d = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    points = []
    max_r = 1.5
    k_max = int(np.ceil(max_r / d)) + 1
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
    elif len(points) < n:
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
    boundary_min = 1.0 - np.max(np.hypot(centers[:, 0], centers[:, 1]))
    
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(boundary_min, min_pair_dist / 2.0)


def _feasibility_violation(x, r, n=N):
    """Compute constraint violations for given radius r. Returns (violation_value, gradient).
    Violations: for each pair i<j, need |c_i - c_j| >= 2r, and for each i, |c_i| <= 1-r.
    We minimize max(0, 2r - d_ij) and max(0, |c_i| - (1-r)).
    """
    centers = x.reshape(n, 2)
    grad = np.zeros((n, 2))
    max_viol = 0.0
    
    # Boundary violations: |c_i| - (1-r)
    for i in range(n):
        cx, cy = centers[i]
        dist = np.hypot(cx, cy)
        viol = dist - (1.0 - r)
        if viol > max_viol:
            max_viol = viol
        if viol > 0:
            if dist > 1e-10:
                grad[i, 0] += cx / dist
                grad[i, 1] += cy / dist
    
    # Pairwise violations: 2r - d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            viol = 2.0 * r - d
            if viol > max_viol:
                max_viol = viol
            if viol > 0 and d > 1e-10:
                grad[i, 0] -= dx / d
                grad[i, 1] -= dy / d
                grad[j, 0] += dx / d
                grad[j, 1] += dy / d
    
    return max_viol, grad.reshape(-1)


def _soft_min_objective_and_grad(x, beta, n=N):
    """Soft-min objective with gradient, using log-sum-exp."""
    centers = x.reshape(n, 2)
    
    # Collect all values: boundary (1 - |c_i|) and pairwise (d_ij / 2)
    boundary_vals = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    
    # Compute pairwise distances
    pair_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            pair_dists[i, j] = pair_dists[j, i] = d
    
    # All values: n boundary + n*(n-1)/2 pairwise
    all_vals = np.concatenate([boundary_vals, pair_dists[np.triu_indices(n, 1)] / 2.0])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * min_val
    soft_min = -log_sum_exp / beta
    
    # Weights for gradient
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft min
    grad = np.zeros((n, 2))
    
    # Boundary terms: d/dx_i (1 - |c_i|) = -c_i / |c_i|
    for i in range(n):
        r = np.hypot(centers[i, 0], centers[i, 1])
        if r > 1e-10:
            grad[i, 0] += weights[i] * (-centers[i, 0] / r)
            grad[i, 1] += weights[i] * (-centers[i, 1] / r)
    
    # Pairwise terms: d/dx_i (d_ij / 2) = (c_i - c_j) / d_ij
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
    return -soft_min, -grad.reshape(-1)


def _try_radius_feasibility(r, x0, deadline, max_iter=500):
    """Try to find a feasible configuration for radius r starting from x0.
    Returns (success, best_centers) if successful."""
    n = N
    
    def obj_and_grad(x):
        viol, grad = _feasibility_violation(x, r, n)
        return viol, grad
    
    try:
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x0.copy(),
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.x is not None:
            final_viol, _ = _feasibility_violation(result.x, r, n)
            if final_viol < 1e-7:
                return True, result.x.reshape(n, 2)
    except Exception:
        pass
    
    return False, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial placement: hexagonal grid
    centers = _initial_hex_grid(N)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Optimize with soft-min to get a good starting point
    x0 = centers.flatten()
    try:
        result = minimize(
            lambda x: _soft_min_objective_and_grad(x, 50.0)[0],
            x0,
            jac=lambda x: _soft_min_objective_and_grad(x, 50.0)[1],
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if result.x is not None:
            new_centers = result.x.reshape(N, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
    except Exception:
        pass
    
    if time.time() > deadline - 30.0:
        return best_centers
    
    # Phase 2: Binary search on radius using feasibility checks
    # Start from current best and try to push higher
    r_low = best_radius
    r_high = min(0.15, 1.0 / np.sqrt(N / np.pi))  # upper bound estimate
    
    # Do binary search for up to 10 iterations or until time runs out
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        success, new_centers = _try_radius_feasibility(r_mid, best_centers.flatten(), deadline)
        
        if success:
            r_low = r_mid
            best_centers = new_centers
            best_radius = _compute_radius(best_centers)
        else:
            r_high = r_mid
    
    # Phase 3: Final polish with tighter soft-min
    if time.time() < deadline - 5.0:
        try:
            x0 = best_centers.flatten()
            result2 = minimize(
                lambda x: _soft_min_objective_and_grad(x, 100.0)[0],
                x0,
                jac=lambda x: _soft_min_objective_and_grad(x, 100.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            if result2.x is not None:
                new_centers = result2.x.reshape(N, 2)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
        except Exception:
            pass
    
    # Phase 4: Local perturbations for final improvement
    if time.time() < deadline - 2.0:
        try:
            for _ in range(30):
                if time.time() > deadline - 2.0:
                    break
                for _ in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    perturbed = best_centers + np.random.normal(0, 0.0005, (N, 2))
                    radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
                    mask = radii > (1.0 - best_radius)
                    if np.any(mask):
                        perturbed[mask] *= ((1.0 - best_radius) / radii[mask])[:, None]
                    
                    r = _compute_radius(perturbed)
                    if r > best_radius:
                        best_centers = perturbed
                        best_radius = r
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
