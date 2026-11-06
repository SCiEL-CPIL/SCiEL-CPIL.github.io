"""Improved packing: L-BFGS-B with soft-min objective and iterative radius targeting."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(centres):
    """Compute the achievable radius for a set of centres."""
    # Wall constraint: distance from each centre to the nearest wall
    # Square is [-0.5, 0.5]^2
    wall_dists = np.minimum(0.5 - np.abs(centres), 0.5)  # min over 4 walls
    wall_min = np.min(wall_dists, axis=1)  # for each centre
    wall_radius = np.min(wall_min)
    
    # Pairwise constraint
    n = len(centres)
    if n > 1:
        # Compute all pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        pair_radius = min_pair_dist / 2.0
    else:
        pair_radius = 0.5
    
    return min(wall_radius, pair_radius)

def _soft_min(values, beta=100.0):
    """Smooth approximation of min using log-sum-exp."""
    # soft-min(x) = -1/beta * log(sum(exp(-beta*x)))
    # For numerical stability:
    m = np.min(values)
    exp_vals = np.exp(-beta * (values - m))
    return m - np.log(np.sum(exp_vals)) / beta

def _objective_and_grad(x, beta=100.0):
    """Objective: minimize negative of soft-min of all constraints.
    Returns (value, gradient)."""
    n = N
    centres = x.reshape(n, 2)
    
    # Wall constraints: for each centre i and each wall, distance
    # We want: min over i of (0.5 - |c_ix|), (0.5 - |c_iy|)
    # Actually, the radius is limited by min(wall_dists, pairwise_dists/2)
    
    # Collect all constraint values that limit the radius:
    # 1. For each centre i: 0.5 - |x_i|, 0.5 - |y_i| (4 per centre = 800)
    # 2. For each pair (i,j): dist(i,j)/2 (N*(N-1)/2 = 19900)
    
    # For efficiency, we'll work with the soft-min of these.
    
    # Wall terms
    wx = 0.5 - np.abs(centres[:, 0])
    wy = 0.5 - np.abs(centres[:, 1])
    
    # Pairwise terms
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Combine all into one array for soft-min
    # To save memory, we can compute soft-min in parts
    all_vals = np.concatenate([wx, wy, pair_terms.ravel()])
    
    # Soft-min
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min_val = m - np.log(np.sum(exp_vals)) / beta
    
    # Objective is -soft_min_val (we want to maximize)
    obj = -soft_min_val
    
    # Gradient computation
    # d(soft_min)/dx = -sum(exp(-beta*(vals-m)) * d(vals)/dx) / sum(exp(-beta*(vals-m)))
    # Actually: soft_min = m - (1/beta)*log(S) where S = sum(exp(-beta*(v-m)))
    # d(soft_min)/d(v_k) = -exp(-beta*(v_k-m)) / S
    # So d(obj)/dx = -d(soft_min)/dx = sum_k [exp(-beta*(v_k-m))/S * d(v_k)/dx]
    
    S = np.sum(exp_vals)
    weights = exp_vals / S  # shape: (num_constraints,)
    
    grad = np.zeros_like(x)
    
    # Wall x: v_k = 0.5 - |x_i| for i in range(N)
    # d(0.5 - |x_i|)/dx_i = -sign(x_i)
    # weights[0:N] correspond to wx
    grad[0::2] += weights[:N] * (-np.sign(centres[:, 0]))
    
    # Wall y: weights[N:2N] correspond to wy
    grad[1::2] += weights[N:2*N] * (-np.sign(centres[:, 1]))
    
    # Pairwise: weights[2N:] correspond to pair_terms
    # pair_terms[i,j] = dist(i,j)/2
    # d(dist(i,j)/2)/d(x_i) = (x_i - x_j) / (2 * dist(i,j))
    # d(dist(i,j)/2)/d(x_j) = (x_j - x_i) / (2 * dist(i,j))
    
    pair_weights = weights[2*N:].reshape(n, n)
    # Only use upper triangle to avoid double counting
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d > 1e-15:
                w = pair_weights[i, j]  # weight for this pair
                # Contribution to centre i
                grad[2*i] += w * (centres[i, 0] - centres[j, 0]) / (2*d)
                grad[2*i+1] += w * (centres[i, 1] - centres[j, 1]) / (2*d)
                # Contribution to centre j
                grad[2*j] += w * (centres[j, 0] - centres[i, 0]) / (2*d)
                grad[2*j+1] += w * (centres[j, 1] - centres[i, 1]) / (2*d)
    
    return obj, grad

def _objective_only(x, beta=100.0):
    """Objective only (for cases where gradient is too slow)."""
    n = N
    centres = x.reshape(n, 2)
    
    wx = 0.5 - np.abs(centres[:, 0])
    wy = 0.5 - np.abs(centres[:, 1])
    
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    all_vals = np.concatenate([wx, wy, pair_terms.ravel()])
    
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min_val = m - np.log(np.sum(exp_vals)) / beta
    
    return -soft_min_val

def _generate_hex_grid():
    """Generate a hexagonal grid starting point."""
    # Approximate hex packing
    # For N=200, use a hexagonal arrangement
    # Start with a reasonable grid and perturb
    
    # Use a square grid as base, then add small random perturbations
    m = int(np.ceil(np.sqrt(N)))
    # Create m x m grid points
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    
    # Take first N points
    grid = np.array([(x, y) for y in ys for x in xs])
    grid = grid[:N]
    
    # Add small random perturbations to break symmetry
    rng = np.random.default_rng(42)
    grid += rng.uniform(-0.005, 0.005, size=grid.shape)
    
    return grid

def _generate_random_grid():
    """Generate a random starting point within the square."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _generate_hex_like_grid():
    """Generate a hexagonal-like starting point."""
    # Hexagonal packing: rows offset
    # For 200 circles, try to make ~14 rows
    n_rows = 14
    n_cols = 15  # 14*15 = 210, take 200
    
    points = []
    row_height = 1.0 / (n_rows * np.sqrt(3)/2)
    col_width = 1.0 / n_cols
    
    for i in range(n_rows):
        y = -0.5 + (i + 0.5) * row_height
        offset = 0 if i % 2 == 0 else col_width / 2
        for j in range(n_cols):
            x = -0.5 + (j + 0.5) * col_width + offset
            if len(points) < N:
                points.append([x, y])
    
    points = np.array(points[:N])
    
    # Add small perturbations
    rng = np.random.default_rng(42)
    points += rng.uniform(-0.003, 0.003, size=points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple starting points and optimization strategies
    best_centres = None
    best_radius = 0
    
    # Starting points to try
    start_points = [
        _generate_hex_grid(),
        _generate_hex_like_grid(),
        _generate_random_grid(),
    ]
    
    # Also try a few more random starts if time permits
    rng = np.random.default_rng(123)
    for _ in range(3):
        start_points.append(rng.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Optimization parameters
    # Start with moderate beta, increase for polish
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for sp_idx, start in enumerate(start_points):
        if time.time() > deadline:
            break
        
        current_best = start.copy()
        current_radius = _compute_radius(current_best)
        
        # Multi-stage optimization with increasing beta
        for beta in betas:
            if time.time() > deadline:
                break
            
            x0 = current_best.ravel()
            
            # Use L-BFGS-B with explicit gradient
            try:
                result = minimize(
                    _objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500},
                )
                
                if result.success or result.fun < _objective_only(x0, beta):
                    new_centres = result.x.reshape(N, 2)
                    new_radius = _compute_radius(new_centres)
                    
                    if new_radius > current_radius:
                        current_best = new_centres
                        current_radius = new_radius
                        
            except Exception:
                # Fall back to no-gradient if gradient fails
                try:
                    result = minimize(
                        _objective_only,
                        x0,
                        args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300},
                    )
                    
                    if result.fun < _objective_only(x0, beta):
                        new_centres = result.x.reshape(N, 2)
                        new_radius = _compute_radius(new_centres)
                        
                        if new_radius > current_radius:
                            current_best = new_centres
                            current_radius = new_radius
                            
                except Exception:
                    pass
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = current_best
    
    # Final polish with high beta if time permits
    if time.time() < deadline - 20 and best_centres is not None:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline:
                break
            
            x0 = best_centres.ravel()
            try:
                result = minimize(
                    _objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800},
                )
                
                if result.fun < _objective_only(x0, beta):
                    new_centres = result.x.reshape(N, 2)
                    new_radius = _compute_radius(new_centres)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centres = new_centres
                        
            except Exception:
                pass
    
    # Ensure result is valid (within bounds)
    if best_centres is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Clip to ensure within square (with small margin)
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
