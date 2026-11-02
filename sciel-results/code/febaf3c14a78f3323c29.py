"""Pack 50 equal circles in a unit square; maximize common radius via gradient ascent."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 50
HALF = 0.5
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _radius(pos):
    """Compute the feasible radius for a given set of centers (side-1 square centered at origin)."""
    pos = np.asarray(pos, dtype=np.float64)
    # Boundary constraint
    r_bound = np.min(HALF - np.abs(pos))
    if r_bound < 0:
        return 0.0
    # Pairwise constraints
    # Compute all pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_bound, r_pair)

def _objective(pos, r_target):
    """Objective: minimize violation to achieve radius >= r_target.
    We want all pairwise distances >= 2*r_target and boundary clearance >= r_target.
    This is a feasibility problem. We minimize the sum of squared violations.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    # Boundary violations
    x = pos[:, 0]
    y = pos[:, 1]
    # Need x >= -HALF + r, x <= HALF - r, y >= -HALF + r, y <= HALF - r
    # Violation = max(0, (-HALF + r) - x) + max(0, x - (HALF - r)) + ...
    lo_x = -HALF + r_target
    hi_x = HALF - r_target
    lo_y = -HALF + r_target
    hi_y = HALF - r_target
    
    viol_x_lo = np.maximum(0.0, lo_x - x)
    viol_x_hi = np.maximum(0.0, x - hi_x)
    viol_y_lo = np.maximum(0.0, lo_y - y)
    viol_y_hi = np.maximum(0.0, y - hi_y)
    
    boundary_viol = np.sum(viol_x_lo**2 + viol_x_hi**2 + viol_y_lo**2 + viol_y_hi**2)
    
    # Pairwise violations
    # dist_ij < 2*r_target is a violation
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    min_dist_sq = (2 * r_target) ** 2
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    dist_sq_iu = dist_sq[iu]
    viol_pair = np.maximum(0.0, min_dist_sq - dist_sq_iu)
    pairwise_viol = np.sum(viol_pair ** 2)
    
    return boundary_viol + pairwise_viol

def _grad_objective(pos, r_target):
    """Gradient of the objective."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    x = pos[:, 0]
    y = pos[:, 1]
    lo_x = -HALF + r_target
    hi_x = HALF - r_target
    lo_y = -HALF + r_target
    hi_y = HALF - r_target
    
    grad = np.zeros_like(pos)
    
    # Boundary gradients
    # d/dx max(0, lo_x - x)^2 = 2*max(0, lo_x - x)*(-1) if lo_x > x else 0
    viol_x_lo = np.maximum(0.0, lo_x - x)
    viol_x_hi = np.maximum(0.0, x - hi_x)
    viol_y_lo = np.maximum(0.0, lo_y - y)
    viol_y_hi = np.maximum(0.0, y - hi_y)
    
    grad[:, 0] += -2.0 * viol_x_lo + 2.0 * viol_x_hi
    grad[:, 1] += -2.0 * viol_y_lo + 2.0 * viol_y_hi
    
    # Pairwise gradients
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    min_dist_sq = (2 * r_target) ** 2
    
    # grad of sum max(0, min_dist_sq - dist_ij^2)^2 for i<j
    # Let f_ij = max(0, min_dist_sq - dist_ij^2)
    # d/dpos_i f_ij^2 = 2 * f_ij * d/dpos_i (min_dist_sq - dist_ij^2) = 2 * f_ij * (2 * (pos_i - pos_j))
    # But only if f_ij > 0
    
    iu = np.triu_indices(N, k=1)
    dist_sq_iu = dist_sq[iu]
    f_iu = np.maximum(0.0, min_dist_sq - dist_sq_iu)
    
    # Accumulate gradient
    # For each pair (i, j) with f_ij > 0:
    # grad[i] += 4 * f_ij * (pos[i] - pos[j])
    # grad[j] += 4 * f_ij * (pos[j] - pos[i]) = -4 * f_ij * (pos[i] - pos[j])
    
    idx_i = iu[0]
    idx_j = iu[1]
    diff_iu = pos[idx_i] - pos[idx_j]
    
    # 4 * f_ij * (pos_i - pos_j)
    term = 4.0 * f_iu[:, None] * diff_iu
    
    # Add to i, subtract from j
    np.add.at(grad, idx_i, term)
    np.add.at(grad, idx_j, -term)
    
    return grad

def _random_start(rng, r_init):
    """Generate a random valid starting configuration for radius r_init."""
    pos = rng.uniform(-HALF + r_init, HALF - r_init, size=(N, 2))
    return pos

def _optimize_radius(pos, r_target, rng):
    """Try to find a configuration with radius >= r_target starting from pos."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2).copy()
    
    # Use SLSQP or trust-constr? Let's try a simple gradient descent with line search
    # Or use scipy's minimize with bounds
    bounds = [(-HALF + r_target, HALF - r_target)] * (2 * N)
    
    try:
        res = minimize(
            _objective,
            pos.flatten(),
            method='L-BFGS-B',
            jac=lambda x: _grad_objective(x, r_target).flatten(),
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success or res.fun < 1e-8:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return None

def _binary_search_radius(pos_init, r_low, r_high, rng, max_iters=20):
    """Binary search for the maximum radius."""
    best_pos = pos_init.copy()
    best_r = r_low
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        r_mid = (r_low + r_high) / 2
        pos_new = _optimize_radius(pos_init, r_mid, rng)
        if pos_new is not None:
            # Verify
            r_actual = _radius(pos_new)
            if r_actual >= r_mid - 1e-10:
                best_pos = pos_new
                best_r = r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        else:
            r_high = r_mid
        if r_high - r_low < 1e-10:
            break
    
    return best_pos, best_r

def construct():
    rng = np.random.default_rng(42)
    
    # Start with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Initial radius
    r_init = _radius(pos)
    
    # Try to improve with gradient ascent on radius
    # Use a simple approach: iteratively increase radius and re-optimize
    best_pos = pos.copy()
    best_r = r_init
    
    # Multiple random restarts for diversity
    for restart in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        if restart == 0:
            pos_start = pos.copy()
        else:
            # Perturb the best solution
            noise = rng.normal(0, 0.01, size=(N, 2))
            pos_start = best_pos + noise
            # Clip to boundary
            pos_start[:, 0] = np.clip(pos_start[:, 0], -HALF + 0.01, HALF - 0.01)
            pos_start[:, 1] = np.clip(pos_start[:, 1], -HALF + 0.01, HALF - 0.01)
        
        # Binary search for radius
        r_low = best_r
        r_high = 0.1  # Upper bound guess
        pos_best, r_best = _binary_search_radius(pos_start, r_low, r_high, rng, max_iters=15)
        
        if r_best > best_r:
            best_r = r_best
            best_pos = pos_best
    
    # Final polish: local optimization at best radius
    if time.time() < DEADLINE - 5:
        r_target = best_r
        # Try to push slightly higher
        for _ in range(10):
            r_try = r_target * 1.001
            pos_new = _optimize_radius(best_pos, r_try, rng)
            if pos_new is not None:
                r_actual = _radius(pos_new)
                if r_actual > best_r:
                    best_r = r_actual
                    best_pos = pos_new
                else:
                    break
            else:
                break
    
    # Ensure validity
    pos_final = np.clip(best_pos, -HALF + best_r, HALF - best_r)
    
    return pos_final
