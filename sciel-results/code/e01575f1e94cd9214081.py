import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _smooth_max(x, eps=1e-4):
    """Smooth approximation of max(x, 0)."""
    return 0.5 * (x + np.sqrt(x**2 + eps**2))

def _packing_violation_and_grad(x, r_target, eps=1e-4):
    """
    Objective: Sum of squared smooth-max violations.
    x: flattened array [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y]
    r_target: fixed radius to test.
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    # Violation: 2*r - dist
    viol_pair = 2.0 * r_target - pair_dist
    sm_pair = _smooth_max(viol_pair, eps)
    obj_pair = np.sum(sm_pair ** 2)
    
    # Wall distances
    d_left = c[:, 0] + HALF
    d_right = HALF - c[:, 0]
    d_bottom = c[:, 1] + HALF
    d_top = HALF - c[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    viol_wall = r_target - wall_dist
    sm_wall = _smooth_max(viol_wall, eps)
    obj_wall = np.sum(sm_wall ** 2)
    
    obj = obj_pair + obj_wall
    
    # Gradients
    # d(sm_max(x)^2)/dx = 2 * sm_max(x) * d(sm_max(x))/dx
    # d(sm_max(x))/dx = 0.5 * (1 + x / sqrt(x^2 + eps^2))
    
    grad = np.zeros_like(x)
    
    # Pair gradients
    # d obj / d c_i = sum_j 2 * sm_pair_ij * d(sm_pair_ij)/d c_i
    # d(sm_pair_ij)/d c_i = 0.5 * (1 + (2r - d_ij) / sqrt((2r - d_ij)^2 + eps^2)) * d(2r - d_ij)/d c_i
    # d(2r - d_ij)/d c_i = - (c_i - c_j) / d_ij
    
    # We only need to compute for i < j and accumulate to both i and j
    # Let's vectorize the gradient calculation for pairs
    
    # sm_pair is length N*(N-1)/2
    # d_sm_dx = 0.5 * (1 + viol_pair / sqrt(viol_pair^2 + eps^2))
    d_sm_dx = 0.5 * (1.0 + viol_pair / np.sqrt(viol_pair**2 + eps**2))
    # d_obj_d_viol = 2 * sm_pair * d_sm_dx
    d_obj_d_viol = 2.0 * sm_pair * d_sm_dx
    
    # d_obj_d_c_i = d_obj_d_viol * (-(c_i - c_j) / d_ij)
    # d_obj_d_c_j = d_obj_d_viol * ((c_i - c_j) / d_ij)
    
    # Reconstruct full matrix for easier indexing? No, just loop or use advanced indexing.
    # For N=50, looping 1225 pairs is fast enough.
    grad_c = np.zeros_like(c)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        d = pair_dist[k]
        if d < 1e-9: continue
        vec = (c[i] - c[j]) / d
        # Gradient contribution to i: -d_obj_d_viol * vec
        # Gradient contribution to j: +d_obj_d_viol * vec
        grad_c[i] -= d_obj_d_viol[k] * vec
        grad_c[j] += d_obj_d_viol[k] * vec
        
    # Wall gradients
    # d(sm_wall(x)^2)/dx = 2 * sm_wall * d(sm_wall)/dx
    # d(sm_wall)/dx = 0.5 * (1 + viol_wall / sqrt(viol_wall^2 + eps^2))
    d_sm_wall_dx = 0.5 * (1.0 + viol_wall / np.sqrt(viol_wall**2 + eps**2))
    d_obj_d_viol_wall = 2.0 * sm_wall * d_sm_wall_dx
    
    # viol_wall = r - w_i
    # d viol_wall / d c_i = - d w_i / d c_i
    
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = d_obj_d_viol_wall[i]
        
        if abs(dl - wmin) < 1e-10:
            # w = x + 0.5, dw/dx = 1
            # d viol/dx = -1
            # d obj/dx = s * (-1)
            grad_c[i, 0] -= s
        if abs(dr - wmin) < 1e-10:
            # w = 0.5 - x, dw/dx = -1
            # d viol/dx = -(-1) = 1
            # d obj/dx = s * (1)
            grad_c[i, 0] += s
        if abs(db - wmin) < 1e-10:
            # w = y + 0.5, dw/dy = 1
            # d viol/dy = -1
            grad_c[i, 1] -= s
        if abs(dt - wmin) < 1e-10:
            # w = 0.5 - y, dw/dy = -1
            # d viol/dy = 1
            grad_c[i, 1] += s

    grad = grad_c.ravel()
    return float(obj), grad

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def _optimize_for_radius(r_target, P_init, deadline):
    """
    Attempt to find a valid configuration for the given radius r_target.
    Returns (success, best_pts, min_violation)
    """
    x0 = P_init.ravel()
    
    def obj_func(x):
        return _packing_violation_and_grad(x, r_target)[0]
    
    def jac_func(x):
        return _packing_violation_and_grad(x, r_target)[1]
        
    try:
        res = minimize(
            obj_func,
            x0,
            jac=jac_func,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        
        x_opt = res.x
        c_opt = x_opt.reshape(N, 2)
        c_opt = np.clip(c_opt, -HALF, HALF)
        
        # Check final violation
        viol, _ = _packing_violation_and_grad(c_opt.ravel(), r_target)
        
        # Also check actual feasible radius
        r_actual = _compute_r(c_opt)
        
        # If actual radius is >= r_target - epsilon, consider it a success
        if r_actual >= r_target - 1e-6:
            return True, c_opt, viol
        else:
            return False, c_opt, viol
            
    except Exception as e:
        return False, P_init, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Initializations to try for each radius
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.005),
        _grid_init(rng, jitter=0.005),
        _random_init(rng),
    ]
    
    # Binary search for radius
    # Known best is ~0.071377
    # We will search from 0.071 to 0.072
    low = 0.0710
    high = 0.0720
    
    # First, try to get a valid solution at a lower radius to have a starting point
    # Then binary search upwards
    
    # We will iterate through a set of candidate radii in descending order of likelihood?
    # Or just binary search.
    # Since the search space is complex, we might miss the solution if we jump too high.
    # Let's do a coarse-to-fine search.
    
    # Step 1: Coarse search
    candidates = np.linspace(0.0715, 0.0718, 5)
    
    # We will keep the best valid configuration found
    # And if we find a valid config for r, we can try to optimize it further or use it as init for higher r
    
    # Let's try a few radii
    radii_to_try = [0.0713, 0.0714, 0.0715, 0.0716, 0.0717, 0.0718]
    
    # Sort descending to find max quickly?
    # But if we fail at high, we might not have a good init for lower.
    # Let's start from a known feasible radius and go up.
    
    # Start with 0.0712
    current_r = 0.0712
    
    # Find a valid config for current_r
    valid_config = None
    for init in inits:
        if time.time() > deadline - 30: break
        success, pts, viol = _optimize_for_radius(current_r, init, deadline)
        if success:
            valid_config = pts
            r_check = _compute_r(pts)
            if r_check > best_r:
                best_r = r_check
                best_pts = pts
            break
            
    if valid_config is None:
        # If we couldn't even get 0.0712, return best effort from init
        for init in inits:
            r_check = _compute_r(init)
            if r_check > best_r:
                best_r = r_check
                best_pts = init
        return best_pts if best_pts is not None else _hex_init(rng)

    # Now try to increase radius
    # We will try to push the radius up in small steps
    step = 0.0001
    
    while time.time() < deadline - 10:
        next_r = current_r + step
        success = False
        best_pts_for_r = None
        
        # Try multiple inits based on current best config + jitter
        # And also the original inits
        inits_to_try = [
            valid_config + rng.normal(0, 0.001, size=(N, 2)),
            valid_config + rng.normal(0, 0.005, size=(N, 2)),
            valid_config
        ]
        
        for init in inits_to_try:
            if time.time() > deadline - 10: break
            # Clip init to ensure validity for optimizer start
            init = np.clip(init, -HALF + 0.01, HALF - 0.01)
            success, pts, viol = _optimize_for_radius(next_r, init, deadline)
            if success:
                valid_config = pts
                r_check = _compute_r(pts)
                if r_check > best_r:
                    best_r = r_check
                    best_pts = pts
                success = True
                break
                
        if success:
            current_r = next_r
        else:
            # If we failed to increase, try a smaller step or stop
            # Try a smaller step
            step *= 0.5
            if step < 1e-6:
                break
            # Also, maybe try a different init strategy?
            # Let's just break if we can't make progress
            break
            
    # Final polish on the best configuration found
    if best_pts is not None and time.time() < deadline - 2:
        # Try to optimize the current best_pts to see if we can get a slightly higher r
        # by minimizing violation for r = best_r + 0.0001
        target_r = best_r + 0.0001
        success, pts, viol = _optimize_for_radius(target_r, best_pts, deadline)
        if success:
            r_check = _compute_r(pts)
            if r_check > best_r:
                best_r = r_check
                best_pts = pts

    return best_pts
