import numpy as np
from scipy.optimize import minimize, basinhopping
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    # Numerical stability: subtract min_a before exp
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - log_sum / beta

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
    # Clip to ensure non-negative
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    
    # Enforce u + v <= 1 by scaling
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    # Cartesian: x = v * 1 + (1-u-v) * 0.5, y = (1-u-v) * H
    # Vertices: A(0,0), B(1,0), C(0.5, H)
    # Barycentric (w, u, v) corresponding to A, B, C
    # x = w*0 + u*1 + v*0.5 = u + 0.5v
    # y = w*0 + u*0 + v*H = vH
    # Here bary[:,0] is w (weight of A), bary[:,1] is u (weight of B)? 
    # Let's stick to the previous definition in the code:
    # bary[m,0] = 1.0 - x - y / (2.0 * H)  -> w
    # bary[m,1] = x - y / (2.0 * H)        -> u
    # So bary[:,0] is w, bary[:,1] is u. Where is v?
    # v = 1 - w - u
    # In _bary_to_cart:
    # pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    # Wait, the previous code used:
    # pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    # pts[:, 1] = (1.0 - u - v) * H
    # This implies bary[:,0] is u (coeff of B?) and bary[:,1] is v (coeff of C?)
    # Let's re-read _cart_to_bary:
    # bary[m, 0] = 1.0 - x - y / (2.0 * H)  # This is w (coeff of A)
    # bary[m, 1] = x - y / (2.0 * H)        # This is u (coeff of B)
    # So bary[:,0] = w, bary[:,1] = u.
    # v = 1 - w - u.
    
    # Previous _bary_to_cart:
    # u = bary[:,0] (w)
    # v = bary[:,1] (u)
    # s = u+v
    # pts[:,0] = v * 1.0 + (1.0 - u - v) * 0.5
    # This formula: x = u_B * 1 + w_A * 0.5 ? No.
    # x = w_A*0 + u_B*1 + v_C*0.5
    # y = w_A*0 + u_B*0 + v_C*H
    # If bary[:,0] is w_A and bary[:,1] is u_B:
    # v_C = 1 - w_A - u_B
    # x = u_B + 0.5 * v_C = u_B + 0.5 * (1 - w_A - u_B) = 0.5*u_B + 0.5 - 0.5*w_A
    # y = H * v_C = H * (1 - w_A - u_B)
    
    # Let's check the previous _bary_to_cart code:
    # pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    # If u=bary[:,0] (w) and v=bary[:,1] (u_B):
    # x = u_B + (1 - w - u_B)*0.5 = u_B + 0.5 - 0.5w - 0.5u_B = 0.5u_B + 0.5 - 0.5w. Correct.
    # pts[:, 1] = (1.0 - u - v) * H = (1 - w - u_B)*H. Correct.
    
    # So we maintain this convention.
    w = u
    u_B = v
    v_C = 1.0 - w - u_B
    pts[:, 0] = u_B + 0.5 * v_C
    pts[:, 1] = H * v_C
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (w, u_B)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        # y = H * v_C => v_C = y/H
        # x = u_B + 0.5 v_C => u_B = x - 0.5 v_C = x - 0.5 y/H
        # w = 1 - u_B - v_C
        v_C = y / H
        u_B = x - 0.5 * v_C
        w = 1.0 - u_B - v_C
        bary[m, 0] = w
        bary[m, 1] = u_B
    return bary

def _objective_bary(bary_flat, beta=50.0):
    """
    Objective function for optimization in barycentric coordinates.
    Returns negative soft min area.
    """
    bary = bary_flat.reshape(N, 2)
    # Clip to valid range [0, 1] for w and u_B
    # Note: v_C = 1 - w - u_B must also be >= 0 => w + u_B <= 1
    # We handle w+u_B <= 1 in _bary_to_cart by scaling.
    # For optimization, it's better to penalize or project.
    # Let's just project in _bary_to_cart.
    pts = _bary_to_cart(bary)
    return -_soft_min_area(pts, beta)

def _basinhopping_optimize(bary_init, time_limit=20.0, beta=50.0):
    """
    Use Basin Hopping to optimize the soft min area in barycentric space.
    """
    start_t = time.time()
    x0 = bary_init.flatten()
    
    # Local minimizer: Nelder-Mead is robust to non-smoothness, but L-BFGS-B is faster if smooth.
    # Soft-min is smooth. Let's use L-BFGS-B with bounds.
    # Bounds: w in [0, 1], u_B in [0, 1].
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def local_minimizer(x):
        if time.time() - start_t > time_limit:
            return minimize(_objective_bary, x, method='Nelder-Mead', args=(beta,), options={'maxiter': 100})
        return minimize(_objective_bary, x, method='L-BFGS-B', args=(beta,), bounds=bounds, options={'maxiter': 100})

    # Basin hopping parameters
    # Step size should be moderate
    step_size = 0.5
    
    result = basinhopping(
        _objective_bary,
        x0,
        minimizer_kwargs={'args': (beta,), 'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 50}},
        niter=10, # Limit iterations to respect time
        stepsize=step_size,
        minimizer=local_minimizer,
        seed=42
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        # Return best found so far? result.x is the best from the last run or the global best?
        # basin hopping returns the best x found.
        pass
    
    new_bary = result.x.reshape(N, 2)
    new_pts = _bary_to_cart(new_bary)
    score = _min_area(new_pts)
    
    # Compare with initial
    init_pts = _bary_to_cart(bary_init)
    init_score = _min_area(init_pts)
    
    if score > init_score:
        return new_pts, score
    else:
        return init_pts, init_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Initial guess: Grid
    positions = [
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
        (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),
        (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),
        (0.3, 0.7), (0.6, 0.7),
    ]
    bary_init = np.empty((N, 2))
    for m, (u, v) in enumerate(positions):
        # u, v here are w, u_B from previous logic?
        # The previous _initial_guess_grid used (u, v) as bary coords directly.
        # Let's assume (u, v) are w, u_B.
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary_init[m, 0] = u
        bary_init[m, 1] = v
        
    initial_pts = _bary_to_cart(bary_init)
    best_points = initial_pts
    best_score = _min_area(initial_pts)
    
    # Run Basin Hopping from initial guess
    if time.time() < deadline - 30:
        pts, score = _basinhopping_optimize(bary_init, time_limit=20.0, beta=50.0)
        if score > best_score:
            best_points = pts
            best_score = score

    # Try a few random restarts with Basin Hopping
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        # Random barycentric start
        bary_rand = np.random.uniform(0, 1, size=(N, 2))
        # Ensure valid
        s = bary_rand[:, 0] + bary_rand[:, 1]
        mask = s > 1.0
        bary_rand[mask, 0] /= s[mask]
        bary_rand[mask, 1] /= s[mask]
        
        # Use different beta for diversity
        beta = 50.0 + 10.0 * i
        
        pts, score = _basinhopping_optimize(bary_rand, time_limit=15.0, beta=beta)
        if score > best_score:
            best_points = pts
            best_score = score
            
    # Final refinement with higher beta and Nelder-Mead on the best point found
    if best_points is not None and time.time() < deadline - 10:
        bary_best = _cart_to_bary(best_points)
        # Use Nelder-Mead for final polish
        x0 = bary_best.flatten()
        result = minimize(
            _objective_bary,
            x0,
            method='Nelder-Mead',
            args=(200.0,), # High beta for sharp min
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        new_bary = result.x.reshape(N, 2)
        new_pts = _bary_to_cart(new_bary)
        new_score = _min_area(new_pts)
        if new_score > best_score:
            best_points = new_pts
            best_score = new_score

    # Return best points
    if best_points is None:
        return _bary_to_cart(bary_init)
        
    return best_points
