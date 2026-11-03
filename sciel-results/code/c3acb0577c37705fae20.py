"""Optimized packing of 50 equal spheres in a unit cube using continuation + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the common radius for a set of centers in [-0.5, 0.5]^3."""
    n = len(centers)
    if n < 2:
        return 0.0
    wall_dists = np.min(np.abs(centers), axis=1)
    wall_radius = np.min(wall_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    pair_radius = min_pair_dist / 2.0
    return min(wall_radius, pair_radius)

def _soft_min_obj_and_grad(centers_flat, beta):
    """
    Compute -soft_min_radius and its gradient.
    soft_min(x) = -log(sum(exp(-beta*x)))/beta
    We minimize -soft_min = log(sum(exp(-beta*x)))/beta
    """
    centers = centers_flat.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances: for each center i, min over 3 axes of |c_ik|
    # Actually wall dist for center i = min(|x_i|, |y_i|, |z_i|)
    abs_c = np.abs(centers)
    wall_dists = np.min(abs_c, axis=1)  # shape (n,)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_radii = dists / 2.0  # shape (n, n)
    
    # Collect all constraint values: wall_dists (n values) + upper triangle of pair_radii (n*(n-1)/2 values)
    pair_idx_i, pair_idx_j = np.triu_indices(n, k=1)
    pair_vals = pair_radii[pair_idx_i, pair_idx_j]
    
    all_vals = np.concatenate([wall_dists, pair_vals])
    m = len(all_vals)
    
    # soft_min = -log(sum(exp(-beta * all_vals)))/beta
    # obj = -soft_min = log(sum(exp(-beta * all_vals)))/beta
    
    # Numerical stability
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    obj = (log_sum + beta * max_val) / beta
    
    # Gradient: d/dx [log(sum(exp(-beta*x_i)))/beta]
    # = (1/beta) * sum_j exp(-beta*x_j) * (-beta * dx_j/dx) / sum_j exp(-beta*x_j)
    # = -sum_j exp(-beta*x_j) * dx_j/dx / sum_j exp(-beta*x_j)
    # where x_j are the constraint values (wall_dists and pair_vals)
    
    # We need d(all_vals)/d(centers_flat)
    # For wall_dists[i] = min(|c_i0|, |c_i1|, |c_i2|)
    # The gradient of min is tricky. Let's use a soft approach or subgradient.
    # Actually, let's use the fact that wall_dist_i = min over k of |c_ik|
    # d(wall_dist_i)/d(c_ik) = sign(c_ik) if |c_ik| = min, else 0
    # But this is non-differentiable. Let's use a smooth approximation.
    
    # Alternative: use the soft-min formulation directly with smooth gradients
    # For now, let's compute the gradient of the log-sum-exp part
    
    # weights = exp_shifted / sum(exp_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # grad_obj = -weights @ d(all_vals)/d(centers_flat)
    # We need to compute d(all_vals)/d(centers_flat)
    
    # For wall_dists[i]: d(wall_dist_i)/d(c_ik) 
    # Use subgradient: pick the axis with minimum |c_ik|
    min_axes = np.argmin(abs_c, axis=1)  # shape (n,)
    wall_grad = np.zeros_like(centers)
    for i in range(n):
        axis = min_axes[i]
        wall_grad[i, axis] = np.sign(centers[i, axis])
    
    # For pair_vals: d(pair_radius_ij)/d(c_ik) = (c_ik - c_jk) / d_ij for k
    # pair_radius_ij = d_ij / 2
    # d(pair_radius_ij)/d(c_ik) = (c_ik - c_jk) / (2 * d_ij)
    
    grad_flat = np.zeros_like(centers_flat)
    
    # Wall contribution
    for i in range(n):
        w = weights[i]  # weight for wall_dist_i
        grad_flat[3*i:3*i+3] -= w * wall_grad[i]
    
    # Pair contribution
    for idx in range(len(pair_idx_i)):
        i, j = pair_idx_i[idx], pair_idx_j[idx]
        w = weights[n + idx]  # weight for pair_val
        d_ij = dists[i, j]
        if d_ij < 1e-12:
            continue
        # d(pair_radius_ij)/d(c_ik) = (c_ik - c_jk) / (2 * d_ij)
        grad_ij = (centers[i] - centers[j]) / (2.0 * d_ij)
        grad_flat[3*i:3*i+3] -= w * grad_ij
        grad_flat[3*j:3*j+3] += w * grad_ij
    
    return obj, grad_flat

def _generate_fcc_start(n):
    """Generate FCC lattice initial positions."""
    # FCC lattice points in unit cube
    # Conventional cell has points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # For a cube of side L, spacing a = L / m
    m = int(np.ceil(n ** (1.0/3.0) / 2.0 ** 0.5))
    m = max(m, 2)
    a = 1.0 / m
    points = []
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(m + 1):
                x = i * a
                y = j * a
                z = k * a
                # FCC positions
                points.append((x, y, z))
                if i < m: points.append((x + 0.5*a, y + 0.5*a, z))
                if i < m: points.append((x + 0.5*a, y, z + 0.5*a))
                if i < m: points.append((x, y + 0.5*a, z + 0.5*a))
    
    points = np.array(points)
    # Center in [-0.5, 0.5]
    points = points - 0.5
    # Shuffle and take n
    np.random.seed(42)
    idx = np.random.permutation(len(points))[:n]
    return points[idx]

def _generate_bcc_start(n):
    """Generate BCC lattice initial positions."""
    m = int(np.ceil(n ** (1.0/3.0) / 1.5 ** 0.5))
    m = max(m, 2)
    a = 1.0 / m
    points = []
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(m + 1):
                x = i * a
                y = j * a
                z = k * a
                points.append((x, y, z))
                if i < m and j < m and k < m:
                    points.append((x + 0.5*a, y + 0.5*a, z + 0.5*a))
    
    points = np.array(points)
    points = points - 0.5
    np.random.seed(42)
    idx = np.random.permutation(len(points))[:n]
    return points[idx]

def _generate_grid_start(n):
    """Generate simple grid initial positions."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _optimize_lbfgs(centers, beta, max_time=20.0):
    """Run L-BFGS-B with analytic gradient."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    def fun_and_grad(x):
        if time.time() - start_time > max_time:
            raise TimeoutError()
        obj, grad = _soft_min_obj_and_grad(x, beta)
        return obj, grad
    
    try:
        result = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
        )
        new_centers = result.x.reshape(-1, 3)
        return new_centers
    except (TimeoutError, Exception):
        return centers

def _optimize_from_initial(centers_init, beta_schedule, max_time_per_beta=20.0):
    """Run continuation optimization."""
    centers = centers_init.copy()
    current_radius = _compute_radius(centers)
    
    for beta in beta_schedule:
        centers = _optimize_lbfgs(centers, beta, max_time=max_time_per_beta)
        new_radius = _compute_radius(centers)
        if new_radius > current_radius:
            current_radius = new_radius
    
    return centers, current_radius

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try multiple initializations
    best_centers = None
    best_radius = 0.0
    
    inits = [
        _generate_grid_start(N),
        _generate_fcc_start(N),
        _generate_bcc_start(N),
    ]
    
    # Add a few random starts
    for seed in [0, 1, 2]:
        np.random.seed(seed)
        random_centers = np.random.uniform(-0.45, 0.45, (N, 3))
        inits.append(random_centers)
    
    beta_schedule = [5, 20, 50, 100, 200, 400, 800]
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        centers, radius = _optimize_from_initial(
            init,
            beta_schedule=beta_schedule,
            max_time_per_beta=15.0
        )
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = inits[0]
        best_radius = _compute_radius(best_centers)
    
    # Final polish with very high beta
    if time.time() < deadline - 10.0:
        for beta in [1600, 3200, 6400]:
            if time.time() > deadline - 5.0:
                break
            best_centers = _optimize_lbfgs(best_centers, beta, max_time=10.0)
            new_r = _compute_radius(best_centers)
            if new_r > best_radius:
                best_radius = new_r
    
    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        best_centers = inits[0]
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
