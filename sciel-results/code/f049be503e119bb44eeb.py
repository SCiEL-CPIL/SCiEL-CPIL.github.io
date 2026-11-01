import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing radius for given points."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check boundary
    r_vec = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - r_vec)
    if min_boundary < -1e-9:
        return 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair, min_boundary) / 2.0

def _negative_score(points):
    """Objective function for minimization (negative of packing radius)."""
    r = _compute_score(points)
    return -r

def _soft_min_obj(x, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    x is flattened (N*2,) array of points.
    Returns a smooth approximation of -min(all constraints).
    """
    points = x.reshape(N, 2)
    
    # Boundary constraint: 1 - |c_i|
    r_vec = np.hypot(points[:, 0], points[:, 1])
    boundary_constraints = 1.0 - r_vec
    
    # Pairwise constraints: |c_i - c_j| for i < j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Get upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    pair_constraints = dists[iu]
    
    # Combine all constraints
    all_constraints = np.concatenate([boundary_constraints, pair_constraints])
    
    # Soft min: -logsumexp(-beta * constraints) / beta
    # For large beta, this approximates min(constraints)
    max_val = np.max(all_constraints)
    shifted = all_constraints - max_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = max_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return -soft_min_val  # We want to maximize, so minimize negative

def _hexagonal_init():
    """Generate a hexagonal lattice initialization."""
    # Approximate spacing for 300 circles in unit circle
    # Area per circle ~ pi/300, hex spacing ~ sqrt(2*pi/(N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    points = []
    k = int(1.0 / s) + 2
    sqrt3 = np.sqrt(3.0)
    
    for j in range(-k, k + 1):
        y = j * s * sqrt3 / 2.0
        x_offset = (j % 2) * s / 2.0
        for i in range(-k, k + 1):
            x = (i + x_offset) * s
            if np.hypot(x, y) <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, sort by distance to center and take first N
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind='stable')
        points = points[indices[:N]]
    elif len(points) < N:
        # Pad with random points near center
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.5)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return points[:N]

def _random_init():
    """Generate random initialization inside the circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.9, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _optimize_from_init(points, deadline, beta_start=50.0, beta_end=5000.0):
    """
    Optimize starting from given points using L-BFGS-B with adaptive beta.
    """
    best_r = _compute_score(points)
    best_points = points.copy()
    
    # Phase 1: Coarse optimization with low beta
    x0 = points.flatten()
    
    result = minimize(
        _soft_min_obj, 
        x0, 
        args=(beta_start,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < 0:
        new_points = result.x.reshape(N, 2)
        new_r = _compute_score(new_points)
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
    
    if time.time() > deadline - 30:
        return best_points, best_r
    
    # Phase 2: Medium beta
    x0 = best_points.flatten()
    beta_mid = (beta_start + beta_end) / 2
    
    result = minimize(
        _soft_min_obj,
        x0,
        args=(beta_mid,),
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or result.fun < 0:
        new_points = result.x.reshape(N, 2)
        new_r = _compute_score(new_points)
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
    
    if time.time() > deadline - 15:
        return best_points, best_r
    
    # Phase 3: High beta for fine polishing
    x0 = best_points.flatten()
    
    result = minimize(
        _soft_min_obj,
        x0,
        args=(beta_end,),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    if result.success or result.fun < 0:
        new_points = result.x.reshape(N, 2)
        new_r = _compute_score(new_points)
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
    
    return best_points, best_r

def _force_directed_polish(points, deadline):
    """
    Force-directed method to push overlapping circles apart and keep them inside.
    """
    best_r = _compute_score(points)
    best_points = points.copy()
    
    current_points = points.copy()
    step_size = 0.005
    
    for iter in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Compute current radius
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        r_vec = np.hypot(current_points[:, 0], current_points[:, 1])
        current_r = min(np.min(dists), np.min(1.0 - r_vec)) / 2.0
        
        if current_r > best_r:
            best_r = current_r
            best_points = current_points.copy()
        
        # Compute forces
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion for close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * current_r + 0.01:
                    v = current_points[i] - current_points[j]
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:
                        v_unit = v / norm
                        strength = step_size * (2.0 * current_r + 0.01 - d)
                        forces[i] += v_unit * strength
                        forces[j] -= v_unit * strength
        
        # Boundary force
        for i in range(N):
            r_i = r_vec[i]
            if r_i > 0:
                dist_to_bound = 1.0 - r_i
                if dist_to_bound < 2.0 * current_r + 0.01:
                    v_in = -current_points[i] / r_i
                    strength = step_size * (2.0 * current_r + 0.01 - dist_to_bound) * 5.0
                    forces[i] += v_in * strength
        
        # Update
        current_points += forces
        
        # Project back inside if needed
        r_vec_new = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = r_vec_new > 1.0 - 1e-6
        if np.any(mask):
            current_points[mask] *= (1.0 - 1e-6) / r_vec_new[mask, np.newaxis]
        
        # Cool down
        step_size *= 0.999
        if step_size < 1e-5:
            step_size = 0.002
            current_points += np.random.normal(0, 0.0005, current_points.shape)
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_overall_points = None
    best_overall_r = 0.0
    
    # Multiple initializations
    inits = []
    
    # 1. Hexagonal lattice
    hex_points = _hexagonal_init()
    inits.append(hex_points)
    
    # 2. Random initializations
    for _ in range(3):
        rand_points = _random_init()
        inits.append(rand_points)
    
    # Optimize each initialization
    for init_points in inits:
        if time.time() > deadline - 20:
            break
        
        polished_points, polished_r = _optimize_from_init(init_points, deadline)
        
        if polished_r > best_overall_r:
            best_overall_r = polished_r
            best_overall_points = polished_points.copy()
    
    # Final polish with force-directed method if time permits
    if best_overall_points is not None and time.time() < deadline - 10:
        fd_points, fd_r = _force_directed_polish(best_overall_points, deadline)
        if fd_r > best_overall_r:
            best_overall_r = fd_r
            best_overall_points = fd_points.copy()
    
    # If no valid result, return hexagonal init
    if best_overall_points is None:
        best_overall_points = hex_points
    
    return best_overall_points

# EVOLVE-BLOCK-END
