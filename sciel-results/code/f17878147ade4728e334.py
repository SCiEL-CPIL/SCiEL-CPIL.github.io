import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _penalty_objective(x, target_r, penalty_weight):
    """
    Penalty-based objective: minimize sum of squared violations of constraints.
    x: flattened array of shape (N*3,)
    target_r: the radius we want to achieve
    penalty_weight: weight for the penalty term
    """
    centers = x.reshape(N, 3)
    
    # Wall constraints: 0.5 - |c_ik| >= target_r
    wall_violations = np.maximum(0.0, target_r - (0.5 - np.abs(centers)))
    
    # Pairwise constraints: |c_i - c_j| >= 2*target_r
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_violations = np.maximum(0.0, 2.0 * target_r - dists)
    
    # Sum of squared violations
    obj = np.sum(wall_violations**2) + np.sum(pair_violations**2)
    
    # Add a small regularization to keep points away from boundaries
    boundary_reg = 0.01 * np.sum((0.5 - np.abs(centers))**2)
    
    return obj + boundary_reg

def _penalty_gradient(x, target_r, penalty_weight):
    """
    Analytical gradient of the penalty objective.
    """
    centers = x.reshape(N, 3)
    grad = np.zeros_like(x)
    
    # Wall constraint gradient
    wall_margin = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0.0, target_r - wall_margin)
    
    if np.any(wall_violations > 0):
        # d/dx of (target_r - (0.5 - |x|))^2 = -2*(target_r - (0.5 - |x|)) * sign(x)
        wall_grad = -2.0 * wall_violations * np.sign(centers)
        grad += wall_grad.flatten()
    
    # Pairwise constraint gradient
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_violations = np.maximum(0.0, 2.0 * target_r - dists)
    
    # For each pair (i,j) with violation, gradient contribution:
    # d/dx_i of (2r - |c_i - c_j|)^2 = -2*(2r - |c_i - c_j|) * (c_i - c_j)/|c_i - c_j|
    # d/dx_j of (2r - |c_i - c_j|)^2 = +2*(2r - |c_i - c_j|) * (c_i - c_j)/|c_i - c_j|
    
    for i in range(N):
        for j in range(i+1, N):
            if pair_violations[i, j] > 0:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    dir_vec = diff / dist
                    coeff = 2.0 * pair_violations[i, j]
                    grad[i*3:i*3+3] -= coeff * dir_vec
                    grad[j*3:j*3+3] += coeff * dir_vec
    
    # Boundary regularization gradient
    boundary_grad = -0.02 * (0.5 - np.abs(centers)) * np.sign(centers)
    grad += boundary_grad.flatten()
    
    return grad

def _optimize_penalty(centers, deadline, initial_target_r=None):
    """
    Optimize using penalty method with L-BFGS-B.
    Returns (optimized_centers, achieved_radius).
    """
    x0 = centers.flatten()
    
    # Determine initial target radius
    current_r = _compute_radius(centers)
    if initial_target_r is None:
        target_r = current_r * 1.001  # Slightly above current
    else:
        target_r = initial_target_r
    
    # Run L-BFGS-B optimization
    best_x = x0.copy()
    best_r = current_r
    
    # Try to improve by increasing target radius gradually
    while time.time() < deadline - 2.0 and target_r < 0.12:
        # Set up optimization
        result = minimize(
            _penalty_objective,
            x0,
            args=(target_r, 1.0),
            method='L-BFGS-B',
            jac=_penalty_gradient,
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e-6:
            # Check if we achieved the target radius
            new_centers = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centers)
            
            if new_r >= target_r - 1e-8:
                # Success! Try to increase target
                best_x = result.x.copy()
                best_r = new_r
                target_r = min(target_r * 1.005, 0.12)
                x0 = result.x.copy()
            else:
                # Failed to achieve target, keep best so far
                if new_r > best_r:
                    best_r = new_r
                    best_x = result.x.copy()
                break
        else:
            # Optimization failed
            break
    
    return best_x.reshape(N, 3), best_r

def _local_optimize(centers, deadline):
    """
    More structured local optimization.
    1. Identify the limiting constraint (pair or wall).
    2. Move the involved points away from each other or wall.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    while time.time() < deadline - 1.0:
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Find limiting pair
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Check wall constraint
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        i_wall, j_wall = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        if min_dist / 2.0 < min_wall:
            # Pair is limiting
            # Move i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                dir_i = diff / norm
                dir_j = -diff / norm
            else:
                dir_i = np.random.normal(0, 1, 3)
                dir_i /= np.linalg.norm(dir_i)
                dir_j = -dir_i
                
            step = 0.01
            centers[i] += step * dir_i
            centers[j] += step * dir_j
        else:
            # Wall is limiting
            # Move i_wall away from wall
            k = j_wall
            # Direction is towards center (0,0,0)
            dir = -np.sign(centers[i_wall, k])
            if dir == 0:
                dir = 1.0
            step = 0.01
            centers[i_wall, k] += step * dir
            
        centers = np.clip(centers, -0.5, 0.5)
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    initializations = []
    
    # Grids
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise in [0.0, 0.01, 0.02]:
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            initializations.append(centers)
            
    # Halton
    initializations.append(_build_halton_init())
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    # Optimize each initialization
    # We have limited time. Let's allocate time to each.
    n_inits = len(initializations)
    time_per_init = (_get_deadline() - time.time() - 5.0) / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # First do quick local optimization to get a good starting point
        opt_centers, opt_radius = _local_optimize(centers, time.time() + time_per_init * 0.3)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
        # Then use penalty-based optimization with remaining time
        if time.time() < deadline - 5.0:
            penalty_deadline = min(time.time() + time_per_init * 0.7, deadline - 5.0)
            pen_centers, pen_radius = _optimize_penalty(opt_centers, penalty_deadline)
            
            if pen_radius > best_radius:
                best_radius = pen_radius
                best_centers = pen_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
