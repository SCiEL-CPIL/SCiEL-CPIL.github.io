"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    d = pdist(centers)
    pair_min = np.min(d) / 2.0
    return min(wall_min, pair_min)


def _smooth_energy_and_grad(centers, r_target, temp=0.02):
    """
    Compute smooth energy and gradient for a target radius.
    Energy is 0 if all constraints are satisfied (pairwise dist >= 2*r, wall dist >= r).
    Otherwise, it's a smooth penalty.
    """
    centers = np.asarray(centers)
    n = centers.shape[0]
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0
    # Only use upper triangle to avoid double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists_flat = pair_dists[mask]
    
    all_dists = np.concatenate([wall_dists, pair_dists_flat])
    
    # Energy: sum of soft violations
    # violation_i = max(0, r_target - d_i)
    # smooth version: temp * log(1 + exp((r_target - d_i) / temp))
    violations = r_target - all_dists
    # softplus approximation
    energy = temp * np.sum(np.log1p(np.exp(violations / temp)))
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Wall gradients
    # d(0.5 - |x|)/dx = -sign(x)
    # The violation for wall x of point i is r_target - (0.5 - |x_i|)
    # d(violation)/dx_i = -d(0.5 - |x_i|)/dx_i = sign(x_i)
    # d(energy)/dx_i = sum over violations involving this coordinate
    # For wall x: d(energy)/dx_i = temp * sigmoid((r_target - (0.5-|x_i|))/temp) * sign(x_i)
    # Actually, d(softplus(v))/dx = sigmoid(v/temp) * (1/temp) * d(v)/dx
    # v = r_target - d_i, d(v)/dx_i = -d(d_i)/dx_i
    # For wall: d_i = 0.5 - |x_i|, d(d_i)/dx_i = -sign(x_i)
    # So d(v)/dx_i = sign(x_i)
    # d(energy)/dx_i = sigmoid(v/temp) * sign(x_i)
    
    sx = np.sign(centers[:, 0])
    sy = np.sign(centers[:, 1])
    
    sx_sigmoid = 1.0 / (1.0 + np.exp(-(r_target - dx) / temp))
    sy_sigmoid = 1.0 / (1.0 + np.exp(-(r_target - dy) / temp))
    
    grad[:, 0] += sx_sigmoid * sx
    grad[:, 1] += sy_sigmoid * sy
    
    # Pairwise gradients
    # For pair (i,j), violation = r_target - dist_ij/2
    # d(v)/d(centers_i) = -(centers_i - centers_j) / (2 * dist_ij) * ... wait
    # dist_ij = |centers_i - centers_j|
    # d(dist_ij)/d(centers_i) = (centers_i - centers_j) / dist_ij
    # d(v)/d(centers_i) = -d(dist_ij/2)/d(centers_i) = -(centers_i - centers_j) / (2 * dist_ij)
    # d(energy)/d(centers_i) = sigmoid(v/temp) * d(v)/d(centers_i)
    #                        = sigmoid(v/temp) * -(centers_i - centers_j) / (2 * dist_ij)
    
    dist_full = dist.copy()
    np.fill_diagonal(dist_full, 1.0)  # avoid division by zero
    
    # sigmoid for each pair
    pair_violations = r_target - dist_full / 2.0
    pair_sigmoid = 1.0 / (1.0 + np.exp(-pair_violations / temp))
    
    # Only upper triangle matters, but the gradient is symmetric
    # grad_pair[i] = sum_{j != i} pair_sigmoid[i,j] * -(centers_i - centers_j) / (2 * dist_ij)
    #             = sum_{j != i} pair_sigmoid[i,j] * (centers_j - centers_i) / (2 * dist_ij)
    
    # Vectorized:
    # term[i,j] = (centers_j - centers_i) / (2 * dist_ij)
    # We can compute: term = diff / (2 * dist_full[:, :, np.newaxis])
    # But diff[i,j] = centers_i - centers_j, so we need -(diff) / (2 * dist)
    
    term = -diff / (2.0 * dist_full[:, :, np.newaxis])
    grad_pair = np.sum(pair_sigmoid[:, :, np.newaxis] * term, axis=1)
    
    grad += grad_pair
    
    return energy, grad


def _optimize_for_radius(centers, r_target, max_iter=200):
    """
    Optimize centers to satisfy all constraints for a given target radius.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        c = x.reshape(-1, 2)
        e, g = _smooth_energy_and_grad(c, r_target)
        return e, g.flatten()
    
    res = minimize(
        lambda x: objective_and_grad(x)[0],
        centers.flatten(),
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return res.x.reshape(-1, 2)


def _hex_grid_init(n):
    """Generate a hexagonal grid of n points."""
    # Approximate hexagonal packing
    # For n points, use a roughly square arrangement with hexagonal offsets
    m = int(np.ceil(np.sqrt(n)))
    pts = []
    for row in range(m):
        for col in range(m):
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            # Hexagonal offset: shift odd rows by half spacing
            if row % 2 == 1:
                x += 0.5 / m
            pts.append([x, y])
    pts = np.array(pts[:n])
    return pts


def _random_initial(n, seed=None):
    """Generate random initial points."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _polish_centres(centers, max_iter=300):
    """
    Final polish: optimize the minimum radius directly using a smooth approximation.
    """
    # Use a soft-min of all distances
    def soft_min_obj(x):
        c = x.reshape(-1, 2)
        # Wall distances
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        wall_dists = np.concatenate([dx, dy])
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        pair_dists = dist / 2.0
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists_flat = pair_dists[mask]
        
        all_dists = np.concatenate([wall_dists, pair_dists_flat])
        
        # Soft min: -t * log(sum(exp(-d_i / t)))
        t = 0.005
        m = np.min(all_dists)
        # To avoid underflow, subtract min
        exp_vals = np.exp(-(all_dists - m) / t)
        soft_min_val = m - t * np.log(np.sum(exp_vals))
        return -soft_min_val
    
    def soft_min_grad(x):
        c = x.reshape(-1, 2)
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        wall_dists = np.concatenate([dx, dy])
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        pair_dists = dist / 2.0
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists_flat = pair_dists[mask]
        
        all_dists = np.concatenate([wall_dists, pair_dists_flat])
        
        t = 0.005
        m = np.min(all_dists)
        exp_vals = np.exp(-(all_dists - m) / t)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft_min = weighted average of gradients of individual distances
        grad = np.zeros_like(c)
        
        # Wall gradients
        sx = -np.sign(c[:, 0])
        sy = -np.sign(c[:, 1])
        grad[:, 0] += weights[:N] * sx
        grad[:, 1] += weights[N:2*N] * sy
        
        # Pairwise gradients
        dist_full = dist.copy()
        np.fill_diagonal(dist_full, 1.0)
        
        # weights_full: need to map flat weights back to matrix
        weights_full = np.zeros((N, N))
        weights_full[mask] = weights[2*N:]
        # Symmetrize
        weights_full = (weights_full + weights_full.T) / 2.0
        
        term = diff / (2.0 * dist_full[:, :, np.newaxis])
        grad_pair = np.sum(weights_full[:, :, np.newaxis] * term, axis=1)
        grad += grad_pair
        
        # Objective is -soft_min, so gradient is -grad(soft_min)
        return -grad.flatten()
    
    res = minimize(
        soft_min_obj,
        centers.flatten(),
        jac=soft_min_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return res.x.reshape(-1, 2)


def _iterative_radius_increase(centers, r_start, r_end, max_iter_per_step=100):
    """
    Iteratively increase the target radius and optimize.
    """
    current_r = r_start
    step = (r_end - r_start) / 50.0
    
    while current_r < r_end and _remaining_time() > 5.0:
        # Try to satisfy constraints for current_r
        centers = _optimize_for_radius(centers, current_r, max_iter=max_iter_per_step)
        r_achieved = _compute_radius(centers)
        
        if r_achieved >= current_r - 1e-6:
            # Success, increase radius
            current_r += step
        else:
            # Failure, try smaller step
            step /= 2.0
            if step < 1e-8:
                break
    
    return centers, current_r


def construct():
    best_pts = None
    best_r = 0.0
    
    # Start with hexagonal grid
    pts0 = _hex_grid_init(N)
    r0 = _compute_radius(pts0)
    
    # Iterative radius increase
    r_target = r0 + 0.005
    max_r = 0.037
    
    while r_target < max_r and _remaining_time() > 10.0:
        pts_opt = _optimize_for_radius(pts0, r_target, max_iter=100)
        r_opt = _compute_radius(pts_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()
        
        if r_opt >= r_target - 1e-6:
            # Can increase
            r_target += 0.002
        else:
            # Can't increase, try smaller step
            r_target = r_opt + 0.001
    
    # Polish
    if best_pts is not None and _remaining_time() > 5.0:
        pts_polish = _polish_centres(best_pts, max_iter=200)
        r_polish = _compute_radius(pts_polish)
        if r_polish > best_r:
            best_r = r_polish
            best_pts = pts_polish.copy()
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for i in range(5):
        if _remaining_time() < 5.0:
            break
        
        pts_rand = _random_initial(N, seed=i)
        # Quick optimization
        pts_opt = _optimize_for_radius(pts_rand, best_r, max_iter=50)
        r_opt = _compute_radius(pts_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()
        
        # Polish
        if _remaining_time() > 3.0:
            pts_polish = _polish_centres(pts_opt, max_iter=100)
            r_polish = _compute_radius(pts_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish.copy()
    
    if best_pts is None:
        # Fallback: return hexagonal grid
        best_pts = _hex_grid_init(N)
    
    return best_pts
