"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate diverse initializations
    initializations = _generate_initializations()

    best_r = -1
    best_centers = None

    for centers0 in initializations:
        if time.time() > deadline - 15.0:
            break

        # Phase 1: Fast gradient-based optimization
        centers = _gradient_optimize(centers0, deadline=deadline)
        r1 = _compute_radius(centers)

        # Phase 2: Scipy polish with smooth soft-min objective
        centers_polished = _scipy_polish(centers, deadline=deadline)
        if centers_polished is not None:
            r2 = _compute_radius(centers_polished)
            if r2 > r1:
                centers = centers_polished
                r1 = r2

        # Phase 3: Final high-precision polish
        if time.time() < deadline - 5.0:
            centers_final = _final_polish(centers, deadline=deadline)
            if centers_final is not None:
                r3 = _compute_radius(centers_final)
                if r3 > r1:
                    centers = centers_final
                    r1 = r3

        if r1 > best_r:
            best_r = r1
            best_centers = centers.copy()

    if best_centers is None:
        return initializations[0]

    return best_centers


def _generate_initializations():
    """Generate multiple diverse initial configurations."""
    inits = []

    # Grid-based initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    inits.append(grid.copy())

    # Random initialization with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        rand = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(rand)

    # Hexagonal-like packing initialization
    hex_init = _hexagonal_init()
    inits.append(hex_init)

    # Perturbed grid
    rng = np.random.default_rng(777)
    perturbed_grid = grid.copy()
    perturbed_grid += rng.normal(0, 0.02, size=perturbed_grid.shape)
    # Clip to bounds
    perturbed_grid = np.clip(perturbed_grid, -0.49, 0.49)
    inits.append(perturbed_grid)

    return inits


def _hexagonal_init():
    """Create a hexagonal close-packed initialization."""
    centers = []
    # Approximate spacing for hexagonal packing of 50 circles
    # Try different spacings
    best_spacing = None
    best_count = 0

    for spacing in np.arange(0.06, 0.12, 0.001):
        row_h = spacing * math.sqrt(3) / 2
        y = -0.5 + spacing / 2
        row = 0
        while y <= 0.5 - spacing / 2 and len(centers) < N:
            x_offset = 0 if row % 2 == 0 else spacing / 2
            x = -0.5 + spacing / 2 + x_offset
            while x <= 0.5 - spacing / 2 and len(centers) < N:
                centers.append([x, y])
                x += spacing
            y += row_h
            row += 1

        if len(centers) >= N:
            if best_count < N or (best_count == N and best_spacing is None):
                best_spacing = spacing
                best_count = N
                break
        elif len(centers) > best_count:
            best_count = len(centers)
            best_spacing = spacing

    if len(centers) < N:
        # Fill remaining with random
        rng = np.random.default_rng(999)
        while len(centers) < N:
            centers.append(rng.uniform(-0.45, 0.45, size=2).tolist())
    else:
        centers = centers[:N]

    return np.array(centers)


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=0)
    min_wall = np.min(wall_dists)

    # Pair distances
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = 0.5

    return min(min_wall, min_pair)


def _compute_radius_and_gradient(centers, beta=50.0):
    """Compute soft-min radius and its gradient."""
    n = len(centers)
    c = centers

    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    # Wall distances
    wall_dists = np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ])
    wall_dists = np.min(wall_dists, axis=0)

    # Soft min for pairs
    flat_dists = dists[~np.eye(n, dtype=bool)]
    min_pair_soft = -np.log(np.sum(np.exp(-beta * flat_dists))) / beta

    # Soft min for walls
    min_wall_soft = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta

    # Combine: soft min of both
    all_dists = np.concatenate([flat_dists, wall_dists])
    min_soft = -np.log(np.sum(np.exp(-beta * all_dists))) / beta

    return min_soft


# (replaced by _soft_min_radius_and_grad above)


# (replaced by _soft_min_radius above)


def _gradient_optimize(centers, deadline=None):
    """Fast gradient-based optimization of the minimum radius."""
    centers = centers.copy()
    n = len(centers)
    
    # Use L-BFGS-B with analytical gradient of soft-min objective
    x0 = centers.flatten()
    
    best_r = _compute_radius(centers)
    best_x = x0.copy()
    
    # Try multiple beta values for the soft-min approximation
    for beta in [30.0, 50.0, 80.0, 120.0]:
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        def neg_soft_radius(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            return -_soft_min_radius(c, beta)
        
        def neg_soft_radius_grad(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, grad = _soft_min_radius_and_grad(c, beta)
            return -val, -grad
        
        bounds = [(-0.5, 0.5)] * (2 * n)
        
        try:
            result = minimize(neg_soft_radius, x0, method='L-BFGS-B', 
                             jac=neg_soft_radius_grad, bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-15, 
                                     'gtol': 1e-14, 'maxls': 50})
            polished = result.x.reshape(n, 2)
            r = _compute_radius(polished)
            if r > best_r:
                best_r = r
                best_x = result.x.copy()
                x0 = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(n, 2)


def _soft_min_radius(c, beta=50.0):
    """Compute soft-min of all distances (pairs and walls)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ]), axis=0)
    
    # All distances
    all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], wall_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min


def _soft_min_radius_and_grad(c, beta=50.0):
    """Compute soft-min radius and its gradient analytically."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ]), axis=0)
    
    # All distances
    all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], wall_dists])
    
    # Soft min value
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights for each distance
    weights = exp_vals / sum_exp
    
    # Gradient: d(soft_min)/d(c_i) = -sum_k w_k * d(dist_k)/d(c_i)
    grad = np.zeros((n, 2))
    
    # Pair distance gradients
    # For pair (i,j): d(dist_ij)/d(c_i) = -(c_j - c_i)/dist_ij = (c_i - c_j)/dist_ij
    #                 d(dist_ij)/d(c_j) = -(c_i - c_j)/dist_ij = (c_j - c_i)/dist_ij
    pair_weights = weights[:n * (n - 1) // 2]
    
    # Build pair weight matrix
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = pair_weights[pair_idx]
            d = dists[i, j]
            if d > 1e-12:
                direction = (c[i] - c[j]) / d
                grad[i] -= w * direction
                grad[j] += w * direction
            pair_idx += 1
    
    # Wall distance gradients
    wall_weights = weights[n * (n - 1) // 2:]
    for i in range(n):
        w = wall_weights[i]
        # Wall dist = min(0.5 - |x_i|, 0.5 - |y_i|)
        # Gradient depends on which wall is closer
        dist_x = 0.5 - np.abs(c[i, 0])
        dist_y = 0.5 - np.abs(c[i, 1])
        
        if dist_x <= dist_y:
            # Closest to left/right wall
            if c[i, 0] >= 0:
                grad[i, 0] += w  # moving left increases dist
            else:
                grad[i, 0] -= w  # moving right increases dist
        else:
            # Closest to top/bottom wall
            if c[i, 1] >= 0:
                grad[i, 1] += w  # moving down increases dist
            else:
                grad[i, 1] -= w  # moving up increases dist
    
    return soft_min, grad


def _scipy_polish(centers, deadline=None):
    """Polish using scipy with smooth soft-min objective and analytical gradient."""
    if deadline is not None and time.time() > deadline - 8.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    best_result = None
    best_r = _compute_radius(centers)

    for beta in [50.0, 100.0, 200.0]:
        if deadline is not None and time.time() > deadline - 5.0:
            break

        def objective(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            return -_soft_min_radius(c, beta)

        def objective_grad(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, grad = _soft_min_radius_and_grad(c, beta)
            return -val, -grad

        bounds = [(-0.5, 0.5)] * (2 * n)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=objective_grad,
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14})
            polished = result.x.reshape(n, 2)
            r = _compute_radius(polished)
            if r > best_r:
                best_r = r
                best_result = polished
                x0 = result.x.copy()
        except Exception:
            pass

    if best_result is not None and best_r >= _compute_radius(centers) - 1e-10:
        return best_result

    return centers


def _final_polish(centers, deadline=None):
    """High-precision final polish with very tight tolerances and analytical gradient."""
    if deadline is not None and time.time() > deadline - 3.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    beta = 300.0

    def objective(x):
        c = x.reshape(n, 2)
        c = np.clip(c, -0.5, 0.5)
        return -_soft_min_radius(c, beta)

    def objective_grad(x):
        c = x.reshape(n, 2)
        c = np.clip(c, -0.5, 0.5)
        val, grad = _soft_min_radius_and_grad(c, beta)
        return -val, -grad

    bounds = [(-0.5, 0.5)] * (2 * n)

    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          jac=objective_grad,
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
        polished = result.x.reshape(n, 2)
        r_polished = _compute_radius(polished)
        r_orig = _compute_radius(centers)
        if r_polished >= r_orig - 1e-12:
            return polished
    except Exception:
        pass

    return centers

# EVOLVE-BLOCK-END
