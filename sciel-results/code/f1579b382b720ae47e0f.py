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
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _get_distance_components(c):
    """Get all distance constraints (pairwise and boundary) as a flat array."""
    n = c.shape[0]
    # Pairwise distances / 2
    pair_dists = pdist(c, metric='euclidean') / 2.0
    # Boundary distances
    # 0.5 - |x| and 0.5 - |y|
    # We can just use 0.5 - abs(c) which gives 2N values
    bound_dists = 0.5 - np.abs(c)
    
    # Concatenate
    all_dists = np.concatenate([pair_dists, bound_dists.ravel()])
    return all_dists


def _soft_min_objective(params, beta):
    """Minimize negative soft-min of distances."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    dists = _get_distance_components(c)
    
    # Ensure no negative values (numerical safety)
    dists = np.maximum(dists, 1e-12)
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, shift by min
    d_min = np.min(dists)
    shifted = dists - d_min
    # Clip shifted to prevent exp(-beta * large) underflow or exp(0) issues
    # If beta is large, exp(-beta * shifted) will be 0 for most, 1 for min.
    # This is fine.
    # However, if beta * shifted > 700, exp underflows to 0. That's okay.
    # If beta is small, it's smooth.
    
    # Optimization: if beta is very large, just return -min
    if beta > 10000:
        return -d_min
    
    # Log-sum-exp trick
    # log(sum(exp(-beta * shifted)))
    # = log(sum(exp(-beta * (d - d_min))))
    # = log(exp(-beta*0) + sum(exp(-beta * (d_i - d_min))))
    # = log(1 + sum(exp(-beta * (d_i - d_min))))
    
    # Calculate terms
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
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
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _critical_polish(c, max_time=15.0):
    """Polish by solving the active-set KKT system exactly."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for _ in range(50):
        if time.time() > start_time + max_time - 2:
            break
        
        c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
        r = _compute_radius(c)
        
        # Find active constraints
        n = c.shape[0]
        # Pairwise distances
        dists = pdist(c, metric='euclidean')
        # Boundary distances
        bound_dists = 0.5 - np.abs(c).ravel()
        
        # Active pairs: distance < r * (1 + tol)
        tol = 1e-10
        active_pair_mask = dists < r * (1 + tol)
        
        # Active boundary constraints
        active_bound_mask = bound_dists < r * (1 + tol)
        
        # Build constraint Jacobian
        # For each active pair (i,j): (c_i - c_j) / |c_i - c_j|
        # For each active boundary: sign(c_k)
        
        # Count constraints
        n_active_pairs = np.sum(active_pair_mask)
        n_active_bounds = np.sum(active_bound_mask)
        n_constraints = n_active_pairs + n_active_bounds
        
        if n_constraints < 2:
            break
        
        # Build Jacobian matrix J (n_constraints x 2N)
        J = np.zeros((n_constraints, 2 * N))
        
        # Get indices of active pairs
        iu = np.triu_indices(n, k=1)
        pair_indices = list(zip(iu[0][active_pair_mask], iu[1][active_pair_mask]))
        
        for idx, (i, j) in enumerate(pair_indices):
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                J[idx, 2*i] = unit[0]
                J[idx, 2*i+1] = unit[1]
                J[idx, 2*j] = -unit[0]
                J[idx, 2*j+1] = -unit[1]
        
        # Boundary constraints
        for idx in range(n_active_pairs, n_constraints):
            b_idx = idx - n_active_pairs
            # bound_dists.ravel() gives [0.5-|c0x|, 0.5-|c0y|, 0.5-|c1x|, ...]
            k = b_idx // 2
            dim = b_idx % 2
            sign = np.sign(c[k, dim])
            if sign == 0:
                sign = 1.0
            J[idx, 2*k + dim] = -sign
        
        # Solve for Lagrange multipliers: J @ lambda = 1 (all ones)
        # This is an underdetermined system; use least squares
        try:
            lambda_vec, residuals, rank, sv = np.linalg.lstsq(J, np.ones(n_constraints), rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Check if solution is valid (lambda >= 0 for maximization)
        if np.any(lambda_vec < -1e-8):
            # Remove constraints with negative multipliers
            active_mask = lambda_vec >= -1e-8
            if np.sum(active_mask) < 2:
                break
            # Reduce to active set and retry
            J = J[active_mask]
            try:
                lambda_vec, _, _, _ = np.linalg.lstsq(J, np.ones(len(active_mask)), rcond=None)
            except np.linalg.LinAlgError:
                break
            if np.any(lambda_vec < -1e-8):
                break
        
        # Update positions: move along J^T @ lambda
        grad = J.T @ lambda_vec
        # Normalize
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Try small steps in both directions
        for step_size in [1e-6, 1e-5, 1e-4, 1e-3]:
            c_try = c + step_size * grad.reshape(-1, 2)
            c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
            r_try = _compute_radius(c_try)
            if r_try > r + 1e-12:
                best_c = c_try
                best_r = r_try
                break
        
        if best_r - r > 1e-12:
            continue
        else:
            # No improvement, try opposite direction
            for step_size in [1e-6, 1e-5, 1e-4, 1e-3]:
                c_try = c - step_size * grad.reshape(-1, 2)
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _compute_radius(c_try)
                if r_try > r + 1e-12:
                    best_c = c_try
                    best_r = r_try
                    break
            if best_r - r > 1e-12:
                continue
            else:
                break
    
    return best_c, best_r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with annealed soft-min."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Beta schedule: start smooth, end sharp
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_c = c0.copy()
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
            
        try:
            result = minimize(
                _soft_min_objective,
                current_c.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < -best_r:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new
                    current_c = c_new.copy()
        except Exception:
            pass
            
    # Final polish with high precision on the raw objective (non-smooth)
    if time.time() < start_time + max_time - 5:
        try:
            def _objective_raw(params):
                c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r = _compute_radius(c)
                return -r
            
            result = minimize(
                _objective_raw,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            if result.fun < -best_r:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
    
    # Critical point polish
    if time.time() < start_time + max_time - 1:
        try:
            c_crit, r_crit = _critical_polish(best_c, max_time=max(2.0, min(10.0, start_time + max_time - time.time() - 1)))
            if r_crit > best_r:
                best_c = c_crit
                best_r = r_crit
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Use most of the time on hex optimization with annealing
    if time.time() < deadline - 10:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(110.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # If time remains, try a random restart
    if time.time() < deadline - 25:
        np.random.seed(42)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
            
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(15.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final critical polish on best solution
    if time.time() < deadline - 3:
        try:
            c_final, r_final = _critical_polish(best_c, max_time=min(8.0, deadline - time.time() - 3))
            if r_final > best_radius:
                best_radius = r_final
                best_c = c_final.copy()
        except Exception:
            pass
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
