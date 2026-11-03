"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    params: (N, 3) flattened to (3N,).
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    # We want x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1, r >= 0
    # Violations are: r - x, x + r - 1, r - y, y + r - 1, -r
    
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    nr = -r

    # Helper to add softplus penalty terms
    # Penalty term: penalty_weight * softplus(violation, eps)**2
    # Gradient of softplus(v, eps)**2 wrt v is 2 * softplus(v, eps) * sigmoid(v, eps)
    # Gradient wrt params is chain rule.
    
    def add_penalty(violation, grad_indices, signs, params_slice):
        """
        Adds penalty for violation vector.
        violation: (N,) array
        grad_indices: indices in params array corresponding to variables
        signs: array of same shape as violation, indicating how violation depends on params
                e.g., for lx = r - x, grad wrt x is -1, wrt r is 1.
        """
        sp = _softplus(violation, smooth_eps)
        sg = _sigmoid(violation, smooth_eps)
        
        # d(val)/d(params) += penalty_weight * 2 * sp * sg * (d(violation)/d(params))
        factor = 2.0 * penalty_weight * sp * sg
        
        # Update gradient
        # We need to map the violation gradient back to the full param vector
        # This function is a bit generic, so I'll implement specific updates below for clarity and speed
        pass

    # Left x boundary: lx = r - x
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    factor_lx = 2.0 * penalty_weight * sp_lx * sg_lx
    grad[0::3] += -factor_lx  # d(lx)/dx = -1
    grad[2::3] += factor_lx   # d(lx)/dr = 1
    val += penalty_weight * np.sum(sp_lx**2)

    # Right x boundary: ux = x + r - 1
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    factor_ux = 2.0 * penalty_weight * sp_ux * sg_ux
    grad[0::3] += factor_ux   # d(ux)/dx = 1
    grad[2::3] += factor_ux   # d(ux)/dr = 1
    val += penalty_weight * np.sum(sp_ux**2)

    # Left y boundary: ly = r - y
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    factor_ly = 2.0 * penalty_weight * sp_ly * sg_ly
    grad[1::3] += -factor_ly  # d(ly)/dy = -1
    grad[2::3] += factor_ly   # d(ly)/dr = 1
    val += penalty_weight * np.sum(sp_ly**2)

    # Right y boundary: uy = y + r - 1
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    factor_uy = 2.0 * penalty_weight * sp_uy * sg_uy
    grad[1::3] += factor_uy   # d(uy)/dy = 1
    grad[2::3] += factor_uy   # d(uy)/dr = 1
    val += penalty_weight * np.sum(sp_uy**2)

    # Negative radius: nr = -r
    sp_nr = _softplus(nr, smooth_eps)
    sg_nr = _sigmoid(nr, smooth_eps)
    factor_nr = 2.0 * penalty_weight * sp_nr * sg_nr
    grad[2::3] += -factor_nr  # d(nr)/dr = -1
    val += penalty_weight * np.sum(sp_nr**2)

    # Pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
    # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
    
    factor_ov = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    # Accumulate gradients for x_i, y_i, r_i and x_j, y_j, r_j
    # Using np.add.at for efficient scattering
    
    # x_i
    np.add.at(grad[0::3], idx_i, -factor_ov * dx * inv_dist)
    # x_j
    np.add.at(grad[0::3], idx_j, factor_ov * dx * inv_dist)
    # y_i
    np.add.at(grad[1::3], idx_i, -factor_ov * dy * inv_dist)
    # y_j
    np.add.at(grad[1::3], idx_j, factor_ov * dy * inv_dist)
    # r_i
    np.add.at(grad[2::3], idx_i, factor_ov)
    # r_j
    np.add.at(grad[2::3], idx_j, factor_ov)
    
    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _enforce_validity(params):
    """
    Given params (N, 3), enforce constraints by shrinking radii and clipping positions.
    Returns a valid configuration (N, 3).
    """
    p = params.copy()
    # Clip positions to [0, 1]
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    # Clip radii to [0, 0.5]
    p[:, 2] = np.clip(p[:, 2], 0.0, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 0.0))
    
    # Iteratively resolve overlaps by shrinking radii
    # This is a simple heuristic repair
    for _ in range(100):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-9
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially, or split the shrink
            # Simple: shrink both by half the overlap
            shrink = ov / 2.0
            p[i, 2] = max(p[i, 2] - shrink, 0.0)
            p[j, 2] = max(p[j, 2] - shrink, 0.0)
            
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 0.0))
        
    return p


def _optimize(params_init, time_budget=10.0):
    """
    Run L-BFGS-B optimization on the given initial parameters.
    Returns the optimized parameters.
    """
    start_time = time.time()
    best_params = params_init.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # Multi-stage penalty increase
    penalty_weights = [1000.0, 5000.0, 20000.0]
    
    for pw in penalty_weights:
        if time.time() - start_time > time_budget:
            break
            
        def obj(p):
            v, _ = _objective_grad(p, pw)
            return v
            
        def grad(p):
            _, g = _objective_grad(p, pw)
            return g
            
        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B', 
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14, 
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, pw)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass
            
    return best_params


def construct():
    best_params = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    # 1. Jittered Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side)
             for i in range(side) for j in range(side)]
    init_centers = np.array(cells[:N], dtype=float)
    
    # Initial radii estimate: half the grid spacing
    grid_step = 1.0 / side
    init_radii = np.full(N, grid_step * 0.4)
    
    init_params_list = []
    
    # Start 1: Grid
    p1 = np.column_stack((init_centers, init_radii)).flatten()
    init_params_list.append(p1)
    
    # Start 2: Jittered Grid (small)
    jitter = rng.uniform(-0.02, 0.02, (N, 2))
    c2 = init_centers + jitter
    c2 = np.clip(c2, 0.0, 1.0)
    p2 = np.column_stack((c2, init_radii)).flatten()
    init_params_list.append(p2)
    
    # Start 3: Jittered Grid (medium)
    jitter = rng.uniform(-0.05, 0.05, (N, 2))
    c3 = init_centers + jitter
    c3 = np.clip(c3, 0.0, 1.0)
    p3 = np.column_stack((c3, init_radii)).flatten()
    init_params_list.append(p3)
    
    # Start 4: Random
    c4 = rng.uniform(0.1, 0.9, (N, 2))
    r4 = rng.uniform(0.05, 0.15, N)
    p4 = np.column_stack((c4, r4)).flatten()
    init_params_list.append(p4)
    
    # Start 5: Random with larger radii
    c5 = rng.uniform(0.1, 0.9, (N, 2))
    r5 = rng.uniform(0.08, 0.2, N)
    p5 = np.column_stack((c5, r5)).flatten()
    init_params_list.append(p5)
    
    for i, init_p in enumerate(init_params_list):
        if _elapsed() > _DEADLINE - 15:
            break
            
        # Time budget per optimization run
        # We have ~145s total. 5 restarts. ~25s each is safe.
        # But L-BFGS-B is fast. Let's give it 20s each.
        time_budget = min(20.0, _remaining() - 5.0)
        
        if time_budget <= 0:
            break
            
        optimized_p = _optimize(init_p, time_budget=time_budget)
        
        # Enforce validity to ensure we have a valid score
        valid_p = _enforce_validity(optimized_p.reshape(N, 3))
        
        score = np.sum(valid_p[:, 2])
        
        if score > best_score:
            best_score = score
            best_params = valid_p.copy()
            
        # If we have a very good score, we might stop early
        if best_score > 1.25:
            break
            
    if best_params is None:
        # Fallback: Use the first initial config, enforced
        best_params = _enforce_validity(init_params_list[0].reshape(N, 3))
        
    return best_params
