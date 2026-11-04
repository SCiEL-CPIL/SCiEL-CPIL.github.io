"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _soft_min(x, k=50):
    """
    Compute the soft minimum of an array x.
    Uses log-sum-exp trick for stability.
    """
    # If k is large, it approximates the true min
    # soft_min(x) = -1/k * log(sum(exp(-k*x_i)))
    # To avoid overflow/underflow, subtract min
    x_min = np.min(x)
    # If all are very large, this is fine. If some are very small, exp(-k*x) might be large.
    # Actually, for soft MIN, we want to penalize small values.
    # Let's use the standard soft-min:
    # min(x) ~ -1/k * log(sum(exp(-k*x)))
    # But if x contains negative values (violations), this is tricky.
    # Let's stick to the formulation: r_i = min(constraints).
    # We want to maximize sum(r_i).
    # Let's define the objective to minimize -sum(r_i).
    
    # A more stable soft-min for positive values (distances):
    # If we treat constraints as distances d_j, r = min(d_j).
    # soft_min(d) = -1/k * log(sum(exp(-k*d)))
    # Gradient w.r.t d_j: exp(-k*d_j) / sum(exp(-k*d_i))
    
    # However, if any d_j is negative, the physical radius is 0 (or invalid).
    # The soft min will be negative if the min is negative.
    # This is good for the optimizer: it penalizes overlaps.
    
    # Stability:
    # Let m = min(x).
    # sum(exp(-k*x)) = exp(-k*m) * sum(exp(-k*(x-m)))
    # log(sum(exp(-k*x))) = -k*m + log(sum(exp(-k*(x-m))))
    # soft_min = -1/k * (-k*m + log(...)) = m - 1/k * log(sum(exp(-k*(x-m))))
    
    m = np.min(x)
    # x - m >= 0
    # exp(-k*(x-m)) <= 1
    # sum is between 1 and len(x)
    # log(sum) is safe
    s = np.sum(np.exp(-k * (x - m)))
    return m - (1.0 / k) * np.log(s)

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    return xy

def _objective_and_grad(params, k=100):
    """
    Objective function for minimization.
    params: (2N,) vector [x_1, y_1, ..., x_N, y_N]
    Returns: (value, gradient)
    value = -sum(soft_min(constraints_i))
    
    For each circle i, the maximum valid radius is:
    r_i = min(
        x_i, 1-x_i, y_i, 1-y_i,  # Walls
        min_{j!=i} (dist(i,j) - r_j)  # This is circular!
    )
    
    Wait, the standard formulation for "maximize sum of radii" with disjoint interiors
    usually fixes the radii or optimizes positions for fixed radii.
    Here we optimize both.
    
    A common approach is to define the "capacity" of a position x_i given other circles' positions and radii.
    But if we optimize radii too, it's coupled.
    
    Alternative formulation:
    Minimize -sum(r_i)
    Subject to:
    x_i - r_i >= 0
    1 - x_i - r_i >= 0
    y_i - r_i >= 0
    1 - y_i - r_i >= 0
    dist(i,j) >= r_i + r_j
    
    We can use a penalty method or convert to an unconstrained problem using soft-min for the constraints?
    No, soft-min is for the variable we are maximizing (r_i).
    
    Let's stick to the "Voronoi cell inradius" idea but implemented via smooth optimization.
    Actually, a simpler smooth surrogate for the objective "Sum of Radii" with constraints is hard.
    
    Let's try a different angle:
    Fixed radii? No, we must optimize radii.
    
    Let's use the following smooth objective:
    For each circle i, define the "clearance" C_i as the soft-min of:
    - Wall distances: x_i, 1-x_i, y_i, 1-y_i
    - Inter-circle distances: dist(i,j) - r_j  (This depends on r_j, which is a variable)
    
    This creates a coupled system.
    
    Better approach:
    Optimize positions x_i, y_i and radii r_i.
    Objective: Minimize -sum(r_i) + Penalty(Constraints)
    
    Or, use the property that at optimum, each circle is "stuck" against neighbors or walls.
    
    Let's use a smooth penalty function for the constraints:
    L(x, r) = -sum(r_i) + lambda * sum( max(0, violation)^2 )
    
    Violations:
    1. Wall: max(0, r_i - x_i), max(0, r_i - (1-x_i)), etc.
    2. Pair: max(0, r_i + r_j - dist(i,j))
    
    This is differentiable almost everywhere (kink at 0). We can smooth the max(0, z) with a softplus-like function or just use the fact that L-BFGS handles non-smoothness reasonably well if we start close, or use a smoothed penalty like z^2/(1+z^2) or similar.
    
    Actually, a very effective method for circle packing is to minimize the "overlap" energy while maximizing radius.
    
    Let's try this specific strategy:
    1. Start with a good grid/random init.
    2. Use `scipy.optimize.minimize` with L-BFGS-B.
    3. Variables: x, y, r for all N circles.
    4. Objective: -sum(r) + Penalty
       Penalty = sum( max(0, r_i - x_i)^2 + ... + max(0, r_i + r_j - dist_ij)^2 )
       
    To make it smooth, we can use `max(0, z)` replaced by `0.5 * (z + sqrt(z^2 + eps))`.
    
    Let's implement this.
    """
    n = N
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # Clip r to be positive
    r = np.maximum(r, 1e-6)
    
    obj = -np.sum(r)
    grad_obj = np.zeros_like(params)
    grad_obj[2*n:] = -1.0
    
    # Penalty terms
    eps = 1e-4
    pen = 0.0
    grad_pen = np.zeros_like(params)
    
    # 1. Wall constraints
    # r_i <= x_i  => r_i - x_i <= 0
    # r_i <= 1 - x_i => r_i + x_i - 1 <= 0
    # r_i <= y_i => r_i - y_i <= 0
    # r_i <= 1 - y_i => r_i + y_i - 1 <= 0
    
    for i in range(n):
        # x - r >= 0 => r - x <= 0
        v1 = r[i] - xy[i, 0]
        if v1 > 0:
            # Smooth max(0, v1)
            sv1 = 0.5 * (v1 + np.sqrt(v1*v1 + eps))
            dsdv1 = 0.5 * (1 + v1 / np.sqrt(v1*v1 + eps))
            pen += sv1 * sv1
            dpen_dv1 = 2 * sv1 * dsdv1
            grad_pen[2*n + i] += dpen_dv1  # dr
            grad_pen[i*2] -= dpen_dv1       # dx
            
        # 1 - x - r >= 0 => x + r - 1 <= 0
        v2 = xy[i, 0] + r[i] - 1.0
        if v2 > 0:
            sv2 = 0.5 * (v2 + np.sqrt(v2*v2 + eps))
            dsdv2 = 0.5 * (1 + v2 / np.sqrt(v2*v2 + eps))
            pen += sv2 * sv2
            dpen_dv2 = 2 * sv2 * dsdv2
            grad_pen[i*2] += dpen_dv2       # dx
            grad_pen[2*n + i] += dpen_dv2   # dr
            
        # y - r >= 0 => r - y <= 0
        v3 = r[i] - xy[i, 1]
        if v3 > 0:
            sv3 = 0.5 * (v3 + np.sqrt(v3*v3 + eps))
            dsdv3 = 0.5 * (1 + v3 / np.sqrt(v3*v3 + eps))
            pen += sv3 * sv3
            dpen_dv3 = 2 * sv3 * dsdv3
            grad_pen[2*n + i] += dpen_dv3   # dr
            grad_pen[i*2 + 1] -= dpen_dv3   # dy
            
        # 1 - y - r >= 0 => y + r - 1 <= 0
        v4 = xy[i, 1] + r[i] - 1.0
        if v4 > 0:
            sv4 = 0.5 * (v4 + np.sqrt(v4*v4 + eps))
            dsdv4 = 0.5 * (1 + v4 / np.sqrt(v4*v4 + eps))
            pen += sv4 * sv4
            dpen_dv4 = 2 * sv4 * dsdv4
            grad_pen[i*2 + 1] += dpen_dv4   # dy
            grad_pen[2*n + i] += dpen_dv4   # dr
            
    # 2. Pairwise constraints
    # r_i + r_j <= dist(i, j)
    # v = r_i + r_j - dist(i, j) <= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-8:
                dist = 1e-8
                dx_d_dist = 0
                dy_d_dist = 0
            else:
                dx_d_dist = dx / dist
                dy_d_dist = dy / dist
                
            v = r[i] + r[j] - dist
            if v > 0:
                sv = 0.5 * (v + np.sqrt(v*v + eps))
                dsdv = 0.5 * (1 + v / np.sqrt(v*v + eps))
                pen += sv * sv
                dpen_dv = 2 * sv * dsdv
                
                # d(v)/d(r_i) = 1, d(v)/d(r_j) = 1
                # d(v)/d(x_i) = -dx_d_dist, d(v)/d(x_j) = +dx_d_dist
                # d(v)/d(y_i) = -dy_d_dist, d(v)/d(y_j) = +dy_d_dist
                
                grad_pen[2*n + i] += dpen_dv
                grad_pen[2*n + j] += dpen_dv
                
                grad_pen[i*2] -= dpen_dv * dx_d_dist
                grad_pen[j*2] += dpen_dv * dx_d_dist
                grad_pen[i*2 + 1] -= dpen_dv * dy_d_dist
                grad_pen[j*2 + 1] += dpen_dv * dy_d_dist

    # Scale penalty
    penalty_weight = 100.0
    total_obj = obj + penalty_weight * pen
    total_grad = grad_obj + penalty_weight * grad_pen
    
    return total_obj, total_grad

def _gradient_polish(xy, r, deadline, seed=0):
    """
    Refine the packing using gradient-based optimization.
    """
    n = N
    params = np.zeros(2*n + n)
    params[:2*n] = xy.flatten()
    params[2*n:] = r
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for i in range(n):
        bounds.append((1e-6, 0.5)) # r
        
    start_time = time.time()
    result = None
    
    # Run minimize
    try:
        result = minimize(
            _objective_and_grad,
            params,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception as e:
        pass
    
    if result is not None and result.success:
        best_params = result.x
    else:
        best_params = params
        
    # Extract best params
    best_xy = best_params[:2*n].reshape(n, 2)
    best_r = best_params[2*n:]
    
    # Ensure validity by clipping radii to fit in box and not overlap (hard clip)
    # This step might break differentiability but ensures a valid output
    for i in range(n):
        best_r[i] = min(best_r[i], best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
        best_r[i] = max(best_r[i], 1e-6)
        
    # Local repair: if overlaps exist, shrink radii
    # Simple iterative shrink
    for _ in range(10):
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = best_xy[i, 0] - best_xy[j, 0]
                dy = best_xy[i, 1] - best_xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_r[i] + best_r[j]:
                    valid = False
                    # Shrink both
                    overlap = (best_r[i] + best_r[j] - dist) / 2.0
                    best_r[i] -= overlap
                    best_r[j] -= overlap
        
        if valid:
            break
            
    # Final wall clip
    for i in range(n):
        best_r[i] = min(best_r[i], best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
        
    total, valid = _eval_packing(best_xy, best_r)
    if not valid:
        return xy, r, 0.0, False
        
    return best_xy, best_r, total, valid

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + Gradient Polish
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        init_pos = _grid_init(side)
        xy = init_pos[:, :2].copy()
        r = init_pos[:, 2].copy()
        
        out_xy, out_r, total, valid = _gradient_polish(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + Gradient Polish
    for seed in range(10):
        if time.time() > deadline:
            break
        xy = _random_init(N, seed=seed)
        r = np.full(N, 0.05)
        
        out_xy, out_r, total, valid = _gradient_polish(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Refine best found with more iterations or different start
    if best_valid and best_xy is not None:
        # Try a few perturbations of the best solution
        for perturb_seed in range(5):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(100 + perturb_seed)
            perturbed_xy = best_xy + rng.normal(0, 0.01, size=(N, 2))
            perturbed_r = best_r * (1 + rng.normal(0, 0.05, size=N))
            
            # Clip
            for i in range(N):
                perturbed_xy[i, 0] = np.clip(perturbed_xy[i, 0], 0, 1)
                perturbed_xy[i, 1] = np.clip(perturbed_xy[i, 1], 0, 1)
                perturbed_r[i] = max(1e-6, perturbed_r[i])
                
            out_xy, out_r, total, valid = _gradient_polish(perturbed_xy, perturbed_r, deadline)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
