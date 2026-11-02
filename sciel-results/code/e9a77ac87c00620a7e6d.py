"""Packing 200 equal circles in a unit square using active-set subgradient method."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
import heapq


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


def _get_active_set(c, r, tolerance=1e-6):
    """
    Identify active constraints:
    1. Pairwise distances close to 2r
    2. Boundary distances close to r
    Returns:
        pairs: list of (i, j)
        bounds: list of (i, k) where k is 0 or 1
    """
    pairs = []
    bounds = []
    
    # Pairwise
    dists = pdist(c, metric='euclidean')
    # We want d_ij >= 2r. Active if d_ij <= 2r + tolerance
    active_pairs_idx = np.where(dists <= 2.0 * r + tolerance)[0]
    
    # Map pdist index to (i, j)
    n = c.shape[0]
    for k in active_pairs_idx:
        # Inverse of pdist index mapping
        # k = i*(n-1) - i*(i+1)/2 + (j-i-1)
        # Solve for i
        # 2*k = 2*i*n - 2*i - i^2 - i + 2*j - 2*i - 2
        # This is tricky to invert analytically in O(1) without search, 
        # but for N=200, we can use a lookup or simple loop if sparse.
        # However, pdist is dense. Let's use a helper.
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        # Check if this i is correct
        # k_start = i*(n-1) - i*(i+1)//2
        # j = i + 1 + (k - k_start)
        k_start = i * (n - 1) - i * (i + 1) // 2
        if k < k_start:
             i -= 1
             k_start = i * (n - 1) - i * (i + 1) // 2
        j = i + 1 + (k - k_start)
        pairs.append((i, j))
        
    # Boundary
    boundary_dists = 0.5 - np.abs(c)
    # Active if dist <= r + tolerance
    active_bounds_idx = np.where(boundary_dists <= r + tolerance)
    for i, k in zip(active_bounds_idx[0], active_bounds_idx[1]):
        bounds.append((i, k))
        
    return pairs, bounds


def _subgradient_step(c, r, pairs, bounds, step_size):
    """
    Compute the subgradient of the minimum distance function and take a step.
    We want to maximize r.
    The constraint is min( d_ij/2, b_i ) >= r.
    
    If a pair is active, moving them apart increases r.
    If a boundary is active, moving away from boundary increases r.
    
    The subgradient of -f(x) (where f is the min distance) gives the direction to decrease f, 
    so we want the negative of that to increase f.
    
    Let g be the subgradient of the min-distance function w.r.t coordinates.
    We update c <- c + step_size * g.
    
    For an active pair (i,j) with distance d ~ 2r:
    The term is d/2. Gradient of d w.r.t c_i is (c_i - c_j)/d.
    Gradient of d/2 w.r.t c_i is (c_i - c_j)/(2d).
    Since we want to increase the min, we move in the direction of the gradient of the active terms.
    
    For an active boundary (i,k):
    The term is b_i = 0.5 - |c_ik|.
    Gradient w.r.t c_ik is -sign(c_ik).
    To increase b_i, we move in direction sign(c_ik).
    
    We average the gradients of all active constraints to get a robust direction.
    """
    grad = np.zeros_like(c)
    
    # Pairwise gradients
    if pairs:
        for i, j in pairs:
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                # Random perturbation if coincident
                diff = np.random.normal(0, 1e-6, 2)
                d = np.linalg.norm(diff)
            # Gradient of d/2 w.r.t c_i is diff/(2d)
            # Gradient of d/2 w.r.t c_j is -diff/(2d)
            unit = diff / (2.0 * d)
            grad[i] += unit
            grad[j] -= unit
            
    # Boundary gradients
    if bounds:
        for i, k in bounds:
            # b_i = 0.5 - |c_ik|
            # grad b_i / c_ik = -sign(c_ik)
            # We want to increase b_i, so we add sign(c_ik) to c_ik
            if c[i, k] > 0:
                grad[i, k] += 1.0
            elif c[i, k] < 0:
                grad[i, k] -= 1.0
            else:
                grad[i, k] += np.random.choice([-1.0, 1.0])
                
    # Normalize gradient to have unit norm to control step size
    norm = np.linalg.norm(grad)
    if norm > 1e-12:
        grad /= norm
        
    # Update
    c_new = c + step_size * grad
    # Clip to stay within square (with small margin to avoid boundary issues in next step)
    c_new = np.clip(c_new, -0.4999, 0.4999)
    
    return c_new


def _active_set_optimize(c0, max_time=30.0):
    """
    Optimize using active-set subgradient method.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Initial step size
    # Typical distance between centers is ~0.07. Radius ~0.035.
    # We want to move a small fraction of the gap.
    step_size = 0.01
    min_step_size = 1e-6
    max_iter = 5000
    stagnant_count = 0
    stagnation_threshold = 50
    
    c = c0.copy()
    r = best_r
    
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
            
        # Get active set
        pairs, bounds = _get_active_set(c, r, tolerance=1e-5)
        
        # If no active constraints found (should not happen if r is correct min), 
        # it means r is not the true min or tolerance is too tight.
        if not pairs and not bounds:
             # Recalculate r
             r = _compute_radius(c)
             pairs, bounds = _get_active_set(c, r, tolerance=1e-4)
             if not pairs and not bounds:
                 break
                 
        # Compute subgradient step
        c_new = _subgradient_step(c, r, pairs, bounds, step_size)
        
        # Evaluate new radius
        r_new = _compute_radius(c_new)
        
        if r_new > r + 1e-12:
            # Improvement
            r = r_new
            c = c_new
            stagnant_count = 0
            # Optionally increase step size slightly if improving fast? 
            # Keep it simple: constant step size or decay if stagnant.
        else:
            # No improvement
            stagnant_count += 1
            # Reduce step size
            step_size *= 0.95
            if step_size < min_step_size:
                # Restart with random perturbation
                c = c + np.random.normal(0, 0.001, size=c.shape)
                c = np.clip(c, -0.4999, 0.4999)
                step_size = 0.005
                stagnant_count = 0
                # Re-evaluate
                r = _compute_radius(c)
                
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    return best_c, best_r


def _slsqp_polish(c0, max_time=30.0):
    """
    SLSQP polish using ONLY active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_active_constraints(c_ref, t_ref):
        """Build inequality constraint functions for SLSQP using only active set."""
        constraints = []
        
        # Find active pairs (within 1.1 * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        # We need d_ij >= 2*t. So constraint is d_ij - 2*t >= 0.
        # Active if d_ij is close to 2*t.
        active_pairs_idx = np.where(dists < 2.0 * t_ref * 1.1)[0]
        
        n = c_ref.shape[0]
        def _pdist_to_ij(k, n):
            i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
            k_start = i * (n - 1) - i * (i + 1) // 2
            if k < k_start:
                 i -= 1
                 k_start = i * (n - 1) - i * (i + 1) // 2
            j = i + 1 + (k - k_start)
            return i, j
            
        for k in active_pairs_idx:
            i, j = _pdist_to_ij(k, n)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
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
        
        # Boundary constraints - only active ones
        boundary_dists = 0.5 - np.abs(c_ref)
        active_bounds_idx = np.where(boundary_dists < t_ref * 1.1)
        
        for i, k in zip(active_bounds_idx[0], active_bounds_idx[1]):
            def _bound_fun(x, _i=i, _k=k):
                t = x[0]
                c_val = x[1 + _i*2 + _k]
                return 0.5 - np.abs(c_val) - t
            
            def _bound_jac(x, _i=i, _k=k):
                grad = np.zeros_like(x)
                grad[0] = -1.0
                c_val = x[1 + _i*2 + _k]
                grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        return constraints
    
    def _slsqp_obj(x):
        t = x[0]
        return -t
    
    def _slsqp_grad(x):
        grad = np.zeros_like(x)
        grad[0] = -1.0
        return grad
    
    current_c = c0.copy()
    current_r = best_r
    
    # Run SLSQP for a few iterations, updating the active set each time
    for attempt in range(5):
        if time.time() > start_time + max_time - 2:
            break
        
        try:
            constraints = _build_active_constraints(current_c, current_r)
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            result = minimize(
                _slsqp_obj,
                x0,
                method='SLSQP',
                bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                jac=_slsqp_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
                current_r = r_new
            else:
                # If no improvement, break to avoid cycling
                break
        except Exception:
            break
    
    return best_c, best_r


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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Active Set Subgradient Optimization
    # This is the core structural change.
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 10
        if remaining > 20:
            c_opt, r_opt = _active_set_optimize(best_c, max_time=min(60.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 3: SLSQP Polish with Active Constraints
    # This refines the result from the subgradient method.
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(30.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    
    # Step 4: One more round of subgradient + SLSQP if time permits, 
    # perhaps with a slightly perturbed start to escape local optima.
    if time.time() < deadline - 15:
        c_perturbed = best_c + np.random.normal(0, 0.005, size=best_c.shape)
        c_perturbed = np.clip(c_perturbed, -0.4999, 0.4999)
        
        remaining = deadline - time.time() - 5
        if remaining > 20:
            c_opt2, r_opt2 = _active_set_optimize(c_perturbed, max_time=min(30.0, remaining - 5))
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_c = c_opt2.copy()
                
            if time.time() < deadline - 5:
                c_slsqp2, r_slsqp2 = _slsqp_polish(best_c, max_time=min(10.0, deadline - time.time() - 2))
                if r_slsqp2 > best_radius:
                    best_radius = r_slsqp2
                    best_c = c_slsqp2.copy()

    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
