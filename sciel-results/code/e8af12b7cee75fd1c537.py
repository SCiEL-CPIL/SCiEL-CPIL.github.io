"""Pack 50 equal spheres in a unit cube to maximize the common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum feasible radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: distance from each center to the nearest face
    dist_to_boundary = np.min(CUBE_HALF - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return 0.5 * min(min_boundary, min_pair)

def _initial_placement():
    """Create an initial valid placement using a grid."""
    # Use a 4x4x4 grid = 64 positions, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _soft_min_obj_and_grad(x, r, beta=50.0):
    """
    Soft-min objective: minimize the soft minimum of all constraints.
    Constraints:
      - Boundary: CUBE_HALF - |c_ik| >= r  for all i,k
      - Pairwise: |c_i - c_j| >= 2r  for all i<j
    
    We use a log-sum-exp based soft-min for smoothness.
    """
    centers = x.reshape(N, 3)
    
    # Boundary constraints: dist_i = CUBE_HALF - max_k |c_ik|
    # Actually we want min_k (CUBE_HALF - |c_ik|) >= r
    # So constraint_i = min_k (CUBE_HALF - |c_ik|) - r >= 0
    abs_c = np.abs(centers)  # (N, 3)
    # dist_to_boundary_i = CUBE_HALF - max_k abs_c[i,k]
    dist_to_boundary = CUBE_HALF - np.max(abs_c, axis=1)  # (N,)
    boundary_margin = dist_to_boundary - r  # (N,)
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    # Only consider i < j to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]  # (N*(N-1)/2,)
    pair_margin = pair_dists - 2 * r  # (N*(N-1)/2,)
    
    # Combine all margins into one array
    all_margins = np.concatenate([boundary_margin, pair_margin])
    n_margins = len(all_margins)
    
    # Soft-min via log-sum-exp
    # soft_min = -log(sum(exp(-beta * margin))) / beta
    # To avoid overflow, shift by max
    shifted = -beta * all_margins
    shifted = shifted - np.max(shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(log_sum_exp + np.max(shifted)) / beta
    
    # Gradient
    # d(soft_min)/d(margin_k) = exp(-beta * margin_k) / sum(exp(-beta * margin_j))
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad = np.zeros_like(x)
    
    # Boundary gradient
    # d(boundary_margin_i)/d(centers[i, k]) = -sign(centers[i,k]) for the k that maximizes abs
    # Actually: dist_to_boundary_i = CUBE_HALF - max_k |c_ik|
    # d(dist_to_boundary_i)/d(c_ik) = -sign(c_ik) if |c_ik| is the max
    # We need to be careful with ties; use a soft max approximation
    # For simplicity, use the argmax
    max_k = np.argmax(abs_c, axis=1)  # (N,)
    for i in range(N):
        k = max_k[i]
        if abs_c[i, k] > 1e-12:
            grad[i, k] += weights[i] * (-np.sign(centers[i, k])) / beta
    
    # Pairwise gradient
    # d(pair_margin_ij)/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # d(pair_margin_ij)/d(c_j) = -(c_i - c_j) / |c_i - c_j|
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = centers[i] - centers[j]
        dist_ij = np.linalg.norm(d)
        if dist_ij > 1e-12:
            direction = d / dist_ij
            grad[i] += weights[N + idx] * direction / beta
            grad[j] += weights[N + idx] * (-direction) / beta
    
    return soft_min_val, grad

def _try_fit_radius(r, initial_centers, max_iter=500, deadline=None):
    """
    Try to find a feasible placement for radius r using gradient descent
    on the soft-min objective. Returns (success, centers).
    """
    centers = initial_centers.copy()
    x = centers.flatten()
    
    def objective_and_grad(x):
        val, grad = _soft_min_obj_and_grad(x, r, beta=50.0)
        return val, grad
    
    # Use L-BFGS-B with bounds
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    best_val = np.inf
    best_x = x.copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = objective_and_grad(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # If soft-min is close to 0, we're likely feasible
        if val < -1e-4:
            break
        
        # Gradient descent step with adaptive step size
        # Use a simple backtracking line search
        step_size = 0.001
        for _ in range(20):
            new_x = x - step_size * grad
            # Clip to bounds
            new_x = np.clip(new_x, -CUBE_HALF, CUBE_HALF)
            new_val, _ = objective_and_grad(new_x)
            if new_val < val:
                x = new_x
                break
            step_size *= 0.5
        else:
            # No improvement, try smaller step
            break
    
    centers_out = best_x.reshape(N, 3)
    
    # Check feasibility
    dist_to_boundary = np.min(CUBE_HALF - np.abs(centers_out), axis=1)
    if np.min(dist_to_boundary) < r - 1e-9:
        return False, centers_out
    
    diff = centers_out[:, np.newaxis, :] - centers_out[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 2 * r - 1e-9:
        return False, centers_out
    
    return True, centers_out

def _refine_placement(centers, deadline=None):
    """
    Refine a placement to maximize the radius using local optimization.
    """
    x = centers.flatten()
    current_r = _compute_radius(centers)
    
    # Try to improve by optimizing the soft-min for the current radius
    # and slightly larger radii
    best_centers = centers.copy()
    best_r = current_r
    
    for r_try in np.linspace(current_r, current_r * 1.05, 5):
        if deadline is not None and time.time() > deadline:
            break
        success, new_centers = _try_fit_radius(r_try, centers, max_iter=200, deadline=deadline)
        if success:
            actual_r = _compute_radius(new_centers)
            if actual_r > best_r:
                best_r = actual_r
                best_centers = new_centers.copy()
                centers = new_centers.copy()
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with a good initial placement
    centers = _initial_placement()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Bisection search on radius
    r_low = best_r
    r_high = 0.14  # Slightly above known best ~0.136
    
    # Try several initial points for diversity
    initial_points = [
        _initial_placement(),
    ]
    
    # Add a slightly perturbed version
    np.random.seed(42)
    perturbed = _initial_placement() + np.random.randn(N, 3) * 0.01
    perturbed = np.clip(perturbed, -CUBE_HALF + 0.1, CUBE_HALF - 0.1)
    initial_points.append(perturbed)
    
    # Bisection with multiple restarts
    while time.time() < deadline:
        r_mid = (r_low + r_high) / 2
        if r_mid <= best_r + 1e-12:
            r_low = r_mid
            continue
        
        found = False
        for init in initial_points:
            if time.time() > deadline - 15:
                break
            success, centers_candidate = _try_fit_radius(r_mid, init, max_iter=300, deadline=deadline - 15)
            if success:
                actual_r = _compute_radius(centers_candidate)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centers = centers_candidate.copy()
                r_low = r_mid
                found = True
                break
        
        if not found:
            r_high = r_mid
        
        # Converged?
        if r_high - r_low < 1e-6:
            break
    
    # Final refinement
    if time.time() < deadline - 10:
        best_centers, best_r = _refine_placement(best_centers, deadline=deadline - 5)
    
    final_r = _compute_radius(best_centers)
    if final_r < 1e-10:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
