"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances
    threshold = min_dist * 1.3
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.3
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


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


def _local_optimize(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=10.0):
    """Polish using SLSQP with active-set constraints for exact local optimum."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    # Build active set: pairs within margin of minimum distance + boundary contacts
    margin = 0.004
    
    def _get_active_set(c, r):
        """Find active constraints (pairs and boundaries within margin of r)."""
        tree = cKDTree(c)
        pairs = tree.query_pairs(r + margin)
        
        # Filter to only those actually close to the minimum
        active_pairs = []
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2 * r + margin:
                active_pairs.append((i, j))
        
        # Boundary contacts
        boundary_indices = []
        for i in range(N):
            for k in range(2):
                if c[i, k] >= 0.5 - r - margin:
                    boundary_indices.append((i, k, 1))
                elif c[i, k] <= -0.5 + r + margin:
                    boundary_indices.append((i, k, -1))
        
        return active_pairs, boundary_indices
    
    # Iterative active-set refinement
    for iteration in range(5):
        if time.time() > start_time + max_time - 2:
            break
        
        c = best_c.copy()
        r = best_r
        
        active_pairs, boundary_indices = _get_active_set(c, r)
        
        if len(active_pairs) == 0 and len(boundary_indices) == 0:
            break
        
        # Build SLSQP constraints
        constraints = []
        bounds = [(-0.4999, 0.4999)] * (2 * N)
        
        for i, j in active_pairs:
            def pair_constraint(params, _i=i, _j=j, _r=r):
                return np.linalg.norm(params[2*_i:2*_i+2] - params[2*_j:2*_j+2]) - 2*_r
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
        
        for i, k, sign in boundary_indices:
            def bound_constraint(params, _i=i, _k=k, _sign=sign, _r=r):
                if _sign == 1:
                    return _r - (params[2*_i+_k] + 0.5)
                else:
                    return _r - (-0.5 - params[2*_i+_k])
            constraints.append({'type': 'ineq', 'fun': bound_constraint})
        
        if not constraints:
            break
        
        def objective(params):
            c_tmp = params.reshape(-1, 2)
            return -_compute_radius(c_tmp)
        
        try:
            result = minimize(
                objective,
                c.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-16, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-12:
                best_c = c_new.copy()
                best_r = r_new
            else:
                # No improvement; try with slightly larger margin to capture more constraints
                margin *= 1.5
        except Exception:
            break
    
    return best_c, best_r


def _bundle_method(c0, r0, max_time=15.0):
    """Bundle method for nonsmooth optimization: maintain subgradient information."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    # Bundle of subgradients
    bundle = []
    max_bundle_size = 50
    
    for iteration in range(30):
        if time.time() > start_time + max_time - 2:
            break
        
        c = best_c.copy()
        r = best_r
        
        # Compute subgradient of the objective at current point
        # The objective is -min(min_pair_dist/2, min_boundary)
        # Subgradient comes from the active constraint(s)
        
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 0.002)
        
        # Find the most constraining pair(s)
        active_pairs = []
        min_pair_dist = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_pair_dist:
                min_pair_dist = d
                active_pairs = [(i, j)]
            elif abs(d - min_pair_dist) < 1e-8:
                active_pairs.append((i, j))
        
        # Check boundary constraints
        active_bounds = []
        min_boundary = float('inf')
        for i in range(N):
            for k in range(2):
                bd = 0.5 - abs(c[i, k])
                if bd < min_boundary:
                    min_boundary = bd
                    active_bounds = [(i, k, np.sign(c[i, k]))]
                elif abs(bd - min_boundary) < 1e-8:
                    active_bounds.append((i, k, np.sign(c[i, k])))
        
        # Determine which constraint is active (pair or boundary)
        r_pair = min_pair_dist / 2.0
        r_bound = min_boundary
        
        if r_pair < r_bound:
            # Pair constraint is active
            subgrad = np.zeros(2 * N)
            for i, j in active_pairs:
                diff = c[i] - c[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-12:
                    direction = diff / norm
                    subgrad[2*i] -= direction[0] / 2
                    subgrad[2*i+1] -= direction[1] / 2
                    subgrad[2*j] += direction[0] / 2
                    subgrad[2*j+1] += direction[1] / 2
            # Add to bundle
            bundle.append((c.copy(), subgrad.copy(), -r))
        else:
            # Boundary constraint is active
            subgrad = np.zeros(2 * N)
            for i, k, sign in active_bounds:
                subgrad[2*i+k] = -sign
            bundle.append((c.copy(), subgrad.copy(), -r))
        
        # Trim bundle if too large
        if len(bundle) > max_bundle_size:
            # Remove oldest entries
            bundle = bundle[-max_bundle_size:]
        
        # Solve QP subproblem: find direction d that minimizes
        # max over bundle of (f_k + g_k^T d) subject to ||d|| <= delta
        # Use a simple approach: average the subgradients of active constraints
        
        if len(bundle) > 0:
            # Compute the "closest" point in the bundle
            # Simple approach: move in the direction of the negative average subgradient
            avg_subgrad = np.mean([b[1] for b in bundle], axis=0)
            norm_sg = np.linalg.norm(avg_subgrad)
            
            if norm_sg > 1e-10:
                # Take a step in the direction of improving the objective
                step_size = 0.005
                direction = -avg_subgrad / norm_sg
                c_new = c + step_size * direction
                c_new = np.clip(c_new, -0.4999, 0.4999)
                
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new.copy()
                    best_r = r_new
                else:
                    # Try smaller step
                    step_size = 0.001
                    c_new = c + step_size * direction
                    c_new = np.clip(c_new, -0.4999, 0.4999)
                    r_new = _compute_radius(c_new)
                    if r_new > best_r:
                        best_c = c_new.copy()
                        best_r = r_new
        
        # Check for convergence
        if len(bundle) >= 2:
            # Check if subgradients are close to zero (optimality condition)
            sg_norms = [np.linalg.norm(b[1]) for b in bundle[-5:]]
            if all(n < 1e-4 for n in sg_norms):
                break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin: 2s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 30:
        max_opt_time = min(15.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts
    np.random.seed(42)
    n_restarts = 2
    
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 15
        if remaining > 5:
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: SLSQP active-set polish
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            c_pol, r_pol = _slsqp_polish(best_c, best_radius, max_time=min(12.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 5: Bundle method for nonsmooth refinement
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol, r_pol = _bundle_method(best_c, best_radius, max_time=min(10.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 6: Ultra-fine L-BFGS-B polish with very low temperature
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_pol, r_pol = _local_optimize(best_c, max_time=min(6.0, remaining),
                                           temperature_schedule=[0.0005, 0.0002, 0.0001, 0.00005])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 7: Final SLSQP micro-polish
    if time.time() < deadline - 1:
        remaining = deadline - time.time() - 0.5
        if remaining > 1:
            c_pol, r_pol = _slsqp_polish(best_c, best_radius, max_time=min(3.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END

