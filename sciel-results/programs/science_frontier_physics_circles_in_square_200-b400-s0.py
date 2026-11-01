"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    if n < 2:
        return -np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    # Combine all distance components
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a range of spacings to find the best fit for N=200
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            # Take the first N points, but maybe we should center them better?
            # For now, just take the first N.
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            if result.fun < -best_r + 1e-10:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _get_active_pairs(c, r_target, margin=0.01):
    """
    Get indices of pairs that are close to the constraint boundary.
    We include pairs with distance < 2*r_target + margin.
    """
    n = c.shape[0]
    tree = cKDTree(c)
    # Query points within distance 2*r_target + margin
    # Using query_ball_point is easier for variable radius
    pairs = set()
    dist_limit = 2 * r_target + margin
    
    for i in range(n):
        neighbors = tree.query_ball_point(c[i], dist_limit)
        for j in neighbors:
            if i < j:
                pairs.add((i, j))
    
    return list(pairs)


def _slsqp_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraints.
    Iteratively increases r_target and solves for feasibility.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    # We will try to increase the radius in small steps
    # Step size starts small and can be adjusted
    step = 1e-4
    max_iter = 50
    
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        
        r_target = best_r + step
        
        # Get active pairs for the current configuration and target radius
        # We use a slightly larger margin to ensure we don't miss constraints
        pairs = _get_active_pairs(best_c, r_target, margin=0.005)
        
        if not pairs:
            # If no pairs are active, we are likely limited by the boundary
            # We can just check boundary constraints
            pass
            
        # Define constraints
        # Boundary: 0.5 - |c_i| - r >= 0
        # Pair: ||c_i - c_j|| - 2*r >= 0
        
        n = N
        
        def constraints_func(x):
            c = x.reshape(-1, 2)
            # Boundary
            bd = 0.5 - np.abs(c) - r_target
            # Pairs
            pdist_vals = np.zeros(len(pairs))
            for idx, (i, j) in enumerate(pairs):
                diff = c[i] - c[j]
                pdist_vals[idx] = np.linalg.norm(diff) - 2 * r_target
            return np.concatenate([bd.ravel(), pdist_vals])
        
        def jac_func(x):
            c = x.reshape(-1, 2)
            jac_rows = []
            
            # Boundary Jacobian
            for i in range(n):
                for k in range(2):
                    row = np.zeros(2 * n)
                    row[2 * i + k] = -np.sign(c[i, k])
                    jac_rows.append(row)
            
            # Pair Jacobian
            for idx, (i, j) in enumerate(pairs):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    dist = 1e-12
                row = np.zeros(2 * n)
                row[2 * i:2 * i + 2] = diff / dist
                row[2 * j:2 * j + 2] = -diff / dist
                jac_rows.append(row)
            
            return np.array(jac_rows)
        
        try:
            result = minimize(
                lambda x: 0.0, # Objective is constant, we just want feasibility
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                # Increase step slightly if we made good progress
                step *= 1.1
                if step > 1e-3:
                    step = 1e-3
            else:
                # If we didn't improve, reduce step and try again or break
                step *= 0.5
                if step < 1e-6:
                    break
                    
        except Exception:
            break
            
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with sparse smooth objective
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 40
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Try perturbed starts with shorter optimization
    np.random.seed(42)
    n_trials = 3
    for trial in range(n_trials):
        if time.time() > deadline - 50:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            # Smaller perturbation for later trials
            noise_scale = 0.005 / (trial + 1)
            c_pert = best_c + np.random.normal(0, noise_scale, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 45:
            remaining = deadline - time.time() - 45
            c_opt, r_opt = _local_optimize_smooth(c_pert, max_time=min(20.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # 4. Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
