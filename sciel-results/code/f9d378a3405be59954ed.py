"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


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
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return -min_boundary
    
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
    
    for d in np.linspace(0.055, 0.075, 20):
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


def _slsqp_constraints(c):
    """Compute constraint values for SLSQP: pairwise and boundary distances minus target radius."""
    n = c.shape[0]
    # Boundary constraints: 0.5 - |c_i| - r >= 0 for each i and each coordinate
    boundary_dists = 0.5 - np.abs(c)  # shape (n, 2)
    
    # Pairwise constraints: ||c_i - c_j|| - 2*r >= 0 for i < j
    dists = pdist(c, metric='euclidean')
    pair_dists = dists / 2.0  # half distances
    
    return boundary_dists.ravel(), pair_dists


def _slsqp_optimize(c0, r_target, max_time=20.0):
    """
    SLSQP optimization: maximize radius by ensuring all constraints >= 0
    at the target radius. We minimize 0 subject to constraints.
    Returns (success, new_c, new_r).
    """
    start_time = time.time()
    
    def objective(x):
        return 0.0  # Constant objective - we just want to satisfy constraints
    
    # Build constraint list
    n = c0.shape[0]
    # Precompute pair indices
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def constraints_func(x):
        c = x.reshape(-1, 2)
        # Boundary constraints
        bd = 0.5 - np.abs(c) - r_target
        # Pairwise constraints
        dists = pdist(c, metric='euclidean') / 2.0 - r_target
        return np.concatenate([bd.ravel(), dists])
    
    def jac_func(x):
        c = x.reshape(-1, 2)
        n = c.shape[0]
        jac_rows = []
        
        # Boundary Jacobian: d(0.5 - |c_ik| - r)/dx = -sign(c_ik)
        for i in range(n):
            for k in range(2):
                row = np.zeros(2 * n)
                row[2 * i + k] = -np.sign(c[i, k])
                jac_rows.append(row)
        
        # Pairwise Jacobian: d(||c_i - c_j||/2 - r)/dx
        for idx, (i, j) in enumerate(pair_indices):
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
            objective,
            c0.flatten(),
            method='SLSQP',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
            constraints=[
                {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
            ],
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        if elapsed > max_time:
            return False, c0, r_target
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        
        # Check if we actually achieved at least r_target
        if r_new >= r_target - 1e-10:
            return True, c_new, r_new
        else:
            return False, c_new, r_new
    except Exception:
        return False, c0, r_target


def _high_precision_polish(c0, max_time=15.0):
    """High-precision polish using both SLSQP and smooth L-BFGS-B."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Try SLSQP with current radius as target (should be satisfied)
    if time.time() < start_time + max_time - 3:
        success, c_new, r_new = _slsqp_optimize(best_c, best_r, max_time=max_time - 3)
        if success and r_new > best_r:
            best_c = c_new
            best_r = r_new
    
    # Smooth L-BFGS-B polish with high beta
    if time.time() < start_time + max_time - 1:
        try:
            def obj(params):
                return _smooth_objective(params, 5000.0)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing, then SLSQP refinement."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 10:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # SLSQP refinement to push radius higher
    if time.time() < start_time + max_time - 5:
        # Try to increase radius slightly and see if SLSQP can satisfy constraints
        r_try = best_r * 1.001
        success, c_new, r_new = _slsqp_optimize(best_c, r_try, max_time=max(3.0, max_time - (time.time() - start_time) - 5))
        if success and r_new > best_r:
            best_c = c_new
            best_r = r_new
        
        # Another try with slightly higher target
        if time.time() < start_time + max_time - 2:
            r_try2 = best_r * 1.0005
            success2, c_new2, r_new2 = _slsqp_optimize(best_c, r_try2, max_time=max(2.0, max_time - (time.time() - start_time) - 2))
            if success2 and r_new2 > best_r:
                best_c = c_new2
                best_r = r_new2
    
    # Final high-precision polish
    if time.time() < start_time + max_time - 1:
        c_polished, r_polished = _high_precision_polish(
            best_c, max_time=max(2.0, max_time - (time.time() - start_time) - 1)
        )
        if r_polished > best_r:
            best_c = c_polished
            best_r = r_polished
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Primary optimization with smooth + SLSQP hybrid
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 40
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(60.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try perturbed starts with shorter optimization
    np.random.seed(42)
    n_trials = 2
    for trial in range(n_trials):
        if time.time() > deadline - 40:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            c_pert = best_c + np.random.normal(0, 0.003, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 35:
            remaining = deadline - time.time() - 35
            c_opt, r_opt = _local_optimize_smooth(c_pert, max_time=min(20.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final, r_final = _high_precision_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
