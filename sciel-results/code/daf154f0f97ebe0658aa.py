"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_init(n):
    """Generate hexagonal lattice points inside unit disk, return first n closest to center."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        dist = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2 inside unit circle
        mask = dist <= 1.0 - spacing / 2.0
        points = points[mask]
        dist = dist[mask]
        if len(points) >= n:
            idx = np.argsort(dist, kind='stable')[:n]
            return points[idx].copy()
        spacing *= 0.97

def _compute_radius(pts):
    """Compute the max common radius for given centers."""
    n = len(pts)
    # Distances to boundary
    dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    return min(min_boundary, min_pair)

def _soft_min_logsumexp(arr, beta):
    """Soft minimum via log-sum-exp: approximates min(arr) smoothly."""
    m = np.min(arr)
    return m - np.log(np.sum(np.exp(-beta * (arr - m)))) / beta

def _negative_soft_radius(pts, beta=50.0):
    """Objective: negative soft-min of (boundary distances, half pairwise distances)."""
    n = len(pts)
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    half_dists = dist_matrix / 2.0
    
    # Collect all constraints
    all_vals = np.concatenate([dists_boundary, half_dists.ravel()])
    
    return -_soft_min_logsumexp(all_vals, beta)

def _negative_soft_radius_grad(pts, beta=50.0):
    """Gradient of negative soft radius."""
    n = len(pts)
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    half_dists = dist_matrix / 2.0
    
    all_vals = np.concatenate([dists_boundary, half_dists.ravel()])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    # Boundary part: d/dx_i of (1 - |x_i|) = -x_i/|x_i|
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += weights[i] * (-pts[i] / r)
        # Note: we want gradient of -soft_min, so negate
        # Actually soft_min ≈ m - log(sum(exp(-beta*(arr-m))))/beta
        # d(soft_min)/d(arr_k) = -exp(-beta*(arr_k-m)) / sum(exp(-beta*(arr-m))) = -weights[k]
        # d(-soft_min)/d(arr_k) = weights[k]
        # d(arr_k)/d(pts) gives us the gradient contribution
    
    # Pairwise part: for pair (i,j), half_dist = |x_i - x_j|/2
    # d(half_dist)/d(x_i) = (x_i - x_j) / (2 * |x_i - x_j|)
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[i, j]
            if d > 1e-12:
                w_ij = weights[n + i*n + j] + weights[n + j*n + i]
                direction = (pts[i] - pts[j]) / d
                grad[i] += w_ij * 0.5 * direction
                grad[j] += w_ij * (-0.5 * direction)
    
    # The full gradient of -soft_min w.r.t. pts is sum_k weights[k] * d(arr_k)/d(pts)
    # But we computed contributions with signs already... let me redo carefully.
    
    # soft_min(arr) = m - (1/beta)*log(sum(exp(-beta*(arr_k - m))))
    # d(soft_min)/d(arr_k) = -exp(-beta*(arr_k-m)) / sum(exp(-beta*(arr-m))) = -w_k
    # d(-soft_min)/d(arr_k) = +w_k
    # So grad(-soft_min) = sum_k w_k * d(arr_k)/d(pts)
    
    # Boundary: arr_i = 1 - |p_i|, d(arr_i)/d(p_i) = -p_i/|p_i|
    # Pair (i,j): arr = |p_i-p_j|/2, d(arr)/d(p_i) = (p_i-p_j)/(2|p_i-p_j|), d(arr)/d(p_j) = (p_j-p_i)/(2|p_i-p_j|)
    
    grad = np.zeros_like(pts)
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += weights[i] * (-pts[i] / r)
    
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[i, j]
            if d > 1e-12:
                w_ij = weights[n + i*n + j] + weights[n + j*n + i]
                direction = (pts[i] - pts[j]) / d
                grad[i] += w_ij * 0.5 * direction
                grad[j] += w_ij * 0.5 * (-direction)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s
    
    # Step 1: Hexagonal initialization
    pts = _hex_init(N)
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    if time.time() > deadline - 5.0:
        return best_pts
    
    # Step 2: Gradient-based optimization with soft-min surrogate
    beta = 30.0  # temperature for soft-min
    
    x0 = pts.ravel()
    
    def objective(x):
        p = x.reshape(-1, 2)
        return -_soft_min_logsumexp(
            np.concatenate([
                1.0 - np.hypot(p[:, 0], p[:, 1]),
                (lambda dm: np.concatenate([dm.ravel()]))(
                    np.sqrt(np.sum((p[:, np.newaxis, :] - p[np.newaxis, :, :])**2, axis=2)) / 2.0
                )
            ]), beta)
    
    # Use L-BFGS-B with bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(
        objective, x0, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline - 10.0:
        # Check if we improved
        if result.fun < -best_radius:
            candidate = result.x.reshape(-1, 2)
            r = _compute_radius(candidate)
            if r > best_radius:
                best_pts = candidate
                best_radius = r
        return best_pts
    
    candidate = result.x.reshape(-1, 2)
    r = _compute_radius(candidate)
    if r > best_radius:
        best_pts = candidate
        best_radius = r
    
    # Step 3: Refinement with higher beta
    if time.time() < deadline - 30.0:
        beta2 = 80.0
        x1 = best_pts.ravel()
        
        def objective2(x):
            p = x.reshape(-1, 2)
            dists_b = 1.0 - np.hypot(p[:, 0], p[:, 1])
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dm = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
            np.fill_diagonal(dm, np.inf)
            all_vals = np.concatenate([dists_b, dm.ravel()])
            return -_soft_min_logsumexp(all_vals, beta2)
        
        result2 = minimize(
            objective2, x1, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if time.time() < deadline - 10.0:
            candidate2 = result2.x.reshape(-1, 2)
            r2 = _compute_radius(candidate2)
            if r2 > best_radius:
                best_pts = candidate2
                best_radius = r2
    
    # Final validity check
    final_pts = best_pts
    # Ensure no point is outside unit circle
    dists = np.hypot(final_pts[:, 0], final_pts[:, 1])
    if np.any(dists > 1.0):
        # Clamp to boundary
        final_pts[dists > 1.0] /= np.where(dists[dists > 1.0] > 0, dists[dists > 1.0][:, np.newaxis], 1.0)
    
    return final_pts
# EVOLVE-BLOCK-END
