"""Packing 200 equal circles in a unit square using hybrid soft-min and active-set SLSQP."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

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

def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use log-sum-exp based soft minimum for numerical stability
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min
    threshold = 0.12
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    soft_min_bound = b_min - T * np.log(np.sum(np.exp(-(boundary_dists - b_min) / T)))
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_min = min(r_pair, r_bound)
    soft_min_combined = r_min - T * np.log(np.sum(np.exp(-(np.array([r_pair, r_bound]) - r_min) / T)))
    
    return -soft_min_combined

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a range of hexagonal spacings
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c

def _local_optimize(c0, max_time=30.0):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
        0.003,
        0.001
    ]
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            c_new = np.clip(result.x.reshape(-2, 2), -0.5, 0.5)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r

def _active_set_slsqp(c0, max_time=60.0):
    """
    Refine the packing using an active-set SLSQP approach.
    This solves max r s.t. d_ij >= 2r and boundary constraints.
    It iteratively adds constraints that are close to being active.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    c = c0.copy()
    r = best_r
    
    # Map from pdist index to (i, j)
    def _pdist_to_ij(k, n):
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j

    no_improve_count = 0
    
    for iteration in range(40):
        if time.time() > start_time + max_time - 5:
            break
            
        # Identify active constraints:
        # 1. Pairs with distance < 2r * 1.10 (10% margin to catch near-active)
        # 2. Boundaries with distance < r * 1.10
        dists = pdist(c, metric='euclidean')
        active_pairs = np.where(dists < 2.0 * r * 1.10)[0]
        
        boundary_dists = 0.5 - np.abs(c)
        # boundary_dists shape (N, 2)
        # Flatten to identify active boundaries
        active_boundaries = []
        for i in range(N):
            for k in range(2):
                if boundary_dists[i, k] < r * 1.10:
                    active_boundaries.append((i, k))

        # If no active constraints, we are far from optimum or stuck
        if len(active_pairs) == 0 and len(active_boundaries) == 0:
            break

        # Build constraints for SLSQP
        # Variables: x = [r, c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y]
        # Objective: minimize -r (to maximize r)
        
        constraints = []
        
        # Pairwise constraints: d_ij - 2r >= 0
        for k in active_pairs:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                r_val = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * r_val
            
            def _pair_jac(x, _i=i, _j=j):
                r_val = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        # Boundary constraints: 0.5 - |c_ik| - r >= 0
        for i, k in active_boundaries:
            def _bound_fun(x, _i=i, _k=k):
                r_val = x[0]
                c_val = x[1 + _i*2 + _k]
                return 0.5 - np.abs(c_val) - r_val
            
            def _bound_jac(x, _i=i, _k=k):
                grad = np.zeros_like(x)
                grad[0] = -1.0
                c_val = x[1 + _i*2 + _k]
                # derivative of -|c| is -sign(c)
                grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        # Objective function
        def _slsqp_obj(x):
            return -x[0]
        
        def _slsqp_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -1.0
            return grad
        
        # Initial guess
        x0 = np.concatenate([[r], c.flatten()])
        
        # Bounds
        bounds = [(0.0, 0.5)] + [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _slsqp_obj,
                x0,
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            if result.success:
                r_new = result.x[0]
                c_new = result.x[1:].reshape(-2, 2)
                
                # Validate against ALL constraints (not just active ones)
                # to ensure we didn't violate an inactive constraint
                c_clipped = np.clip(c_new, -0.5, 0.5)
                r_valid = _compute_radius(c_clipped)
                
                # If the valid radius is close to the solved radius, accept it
                if r_valid > best_r + 1e-12:
                    best_r = r_valid
                    best_c = c_clipped.copy()
                    c = c_clipped.copy()
                    r = r_valid
                    no_improve_count = 0
                else:
                    # Try with slightly relaxed margin in next iteration
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        break
                    # Nudge the current solution slightly to escape local optimum
                    c = c + np.random.uniform(-1e-6, 1e-6, size=c.shape)
                    c = np.clip(c, -0.5, 0.5)
                    r = _compute_radius(c)
            else:
                break
        except Exception:
            break
            
    return best_c, best_r

def _final_polish(c0, r0, max_time=10.0):
    """
    Final high-precision polish pass using tight-tolerance SLSQP
    with all near-active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    c = c0.copy()
    r = r0
    
    def _pdist_to_ij(k, n):
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j

    for iteration in range(10):
        if time.time() > start_time + max_time - 1:
            break
        
        # Use tighter margin: 5% for final polish
        dists = pdist(c, metric='euclidean')
        active_pairs = np.where(dists < 2.0 * r * 1.05)[0]
        
        boundary_dists = 0.5 - np.abs(c)
        active_boundaries = []
        for i in range(N):
            for k in range(2):
                if boundary_dists[i, k] < r * 1.05:
                    active_boundaries.append((i, k))
        
        if len(active_pairs) == 0 and len(active_boundaries) == 0:
            break
        
        constraints = []
        
        for k in active_pairs:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                r_val = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * r_val
            
            def _pair_jac(x, _i=i, _j=j):
                r_val = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        for i, k in active_boundaries:
            def _bound_fun(x, _i=i, _k=k):
                r_val = x[0]
                c_val = x[1 + _i*2 + _k]
                return 0.5 - np.abs(c_val) - r_val
            
            def _bound_jac(x, _i=i, _k=k):
                grad = np.zeros_like(x)
                grad[0] = -1.0
                c_val = x[1 + _i*2 + _k]
                grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        def _slsqp_obj(x):
            return -x[0]
        
        def _slsqp_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -1.0
            return grad
        
        x0 = np.concatenate([[r], c.flatten()])
        bounds = [(0.0, 0.5)] + [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _slsqp_obj,
                x0,
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False}
            )
            
            if result.success:
                c_new = result.x[1:].reshape(-2, 2)
                c_clipped = np.clip(c_new, -0.5, 0.5)
                r_valid = _compute_radius(c_clipped)
                
                if r_valid > best_r + 1e-14:
                    best_r = r_valid
                    best_c = c_clipped.copy()
                    c = c_clipped.copy()
                    r = r_valid
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # 1. Initialize
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Soft-min optimization (L-BFGS-B)
    # This phase is fast and gets us close to the optimum
    if time.time() < deadline - 70:
        c_opt, r_opt = _local_optimize(best_c, max_time=min(40.0, deadline - time.time() - 60))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Active Set SLSQP Refinement
    # This phase is rigorous and extracts the final precision
    if time.time() < deadline - 15:
        c_ref, r_ref = _active_set_slsqp(best_c, max_time=min(60.0, deadline - time.time() - 10))
        if r_ref > best_radius:
            best_radius = r_ref
            best_c = c_ref.copy()
    
    # 4. Final high-precision polish
    if time.time() < deadline - 5:
        c_pol, r_pol = _final_polish(best_c, best_radius, max_time=min(10.0, deadline - time.time() - 3))
        if r_pol > best_radius:
            best_radius = r_pol
            best_c = c_pol.copy()
            
    # 5. Final check and cleanup
    if best_c is None:
        best_c = _initialize_hexagonal()
        
    # Ensure strict bounds compliance
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
