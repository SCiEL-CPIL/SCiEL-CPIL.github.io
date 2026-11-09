"""Fifty equal circles in a square – high-precision packing optimizer."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill


def _score_from_centres(c):
    """Compute the exact radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _min_distance_and_indices(c):
    """
    Compute minimum pairwise distance/2 and minimum wall distance.
    Returns (r, pair_indices, wall_index) where pair_indices are (i,j) for the min pair
    and wall_index is (circle_idx, wall_idx) for the min wall.
    """
    c = np.clip(c, -HALF + 1e-15, HALF - 1e-15)
    
    # Pairwise
    d = pdist(c)
    min_pair_half = d.min() / 2.0
    min_pair_idx = np.argmin(d)
    # Convert condensed index to (i, j)
    n = N
    # Inverse of pdist indexing
    ii, jj = np.unravel_index(min_pair_idx, (n, n), order='F')
    if ii > jj:
        ii, jj = jj, ii
    
    # Walls
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    min_wall = wall_dists.min()
    wall_flat_idx = np.argmin(wall_dists)
    wall_circle, wall_side = np.unravel_index(wall_flat_idx, (N, 4))
    
    r = min(min_pair_half, min_wall)
    return r, (int(ii), int(jj)), (int(wall_circle), int(wall_side))


def _epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -t + penalty for violations.
    x = [c_1, ..., c_N, t] where c_i are 2D centers and t is the radius.
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    d = pdist(c) / 2.0
    
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([d, wall_dists_flat])
    
    violations = np.maximum(0.0, t - all_dists)
    
    if violations.size > 0:
        max_v = violations.max()
        if max_v > 1e-15:
            penalty = np.exp(-beta * violations).sum() / beta
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    return -t + penalty


def _negative_smooth_score(c_flat, beta=100.0):
    """Objective to minimise (legacy, kept for compatibility)."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    all_dists = np.concatenate([d, wall_dists_flat])
    
    min_d = all_dists.min()
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    
    return -soft_min


def _slsqp_constraints(x, t_target):
    """
    Hard constraints for SLSQP: all pairwise dist/2 >= t_target and all wall dist >= t_target.
    Returns array of constraint values (all should be >= 0).
    """
    c = x.reshape(N, 2)
    
    d = pdist(c) / 2.0 - t_target
    
    wall_dists = np.stack([
        c[:, 0] + HALF - t_target,
        HALF - c[:, 0] - t_target,
        c[:, 1] + HALF - t_target,
        HALF - c[:, 1] - t_target
    ], axis=1)
    
    return np.concatenate([d, wall_dists.flatten()])


def _slsqp_objective(x, t_target):
    """
    Objective for SLSQP: minimize -t (but t is fixed, so we just return 0 or a constant).
    Actually, we want to maximize t, so we fix t and check feasibility.
    For SLSQP, we minimize a constant (0) subject to constraints.
    But that's not right. We need to maximize t.
    
    Better approach: Use SLSQP to maximize t by treating t as a variable.
    x = [centers, t], objective = -t, constraints as above.
    """
    t = x[2*N]
    return -t


def _slsqp_constraint_func(x):
    """
    Constraint function for SLSQP with t as part of x.
    x = [c_1..c_N, t], constraints: pairwise_dist/2 - t >= 0, wall_dist - t >= 0
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    d = pdist(c) / 2.0 - t
    
    wall_dists = np.stack([
        c[:, 0] + HALF - t,
        HALF - c[:, 0] - t,
        c[:, 1] + HALF - t,
        HALF - c[:, 1] - t
    ], axis=1)
    
    return np.concatenate([d, wall_dists.flatten()])


def _slsqp_jacobian(x):
    """
    Jacobian of the constraint function for SLSQP.
    Returns a matrix of shape (n_constraints, n_vars).
    n_constraints = 1225 + 200 = 1425
    n_vars = 100 + 1 = 101
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    n_cons = N*(N-1)//2 + 4*N  # 1225 + 200 = 1425
    n_vars = 2*N + 1  # 101
    
    jac = np.zeros((n_cons, n_vars))
    
    # Pairwise constraints: for each pair (i,j), constraint = |c_i - c_j|/2 - t
    # d/dx_k of |c_i - c_j|/2 where x_k is coordinate k
    pair_idx = 0
    for i in range(N):
        for j in range(i+1, N):
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-15:
                # Degenerate, use subgradient
                grad_i = np.array([0.5, 0.5])
                grad_j = -grad_i
            else:
                grad_i = diff / (2 * dist)
                grad_j = -grad_i
            
            # x[2*i] = c_i[0], x[2*i+1] = c_i[1]
            jac[pair_idx, 2*i] = grad_i[0]
            jac[pair_idx, 2*i+1] = grad_i[1]
            jac[pair_idx, 2*j] = grad_j[0]
            jac[pair_idx, 2*j+1] = grad_j[1]
            # d/dt = -1
            jac[pair_idx, 2*N] = -1.0
            pair_idx += 1
    
    # Wall constraints
    # For circle i, wall 0: c_i[0] + 0.5 - t
    # wall 1: 0.5 - c_i[0] - t
    # wall 2: c_i[1] + 0.5 - t
    # wall 3: 0.5 - c_i[1] - t
    wall_idx = N*(N-1)//2
    for i in range(N):
        # Wall 0: x + 0.5 - t
        jac[wall_idx, 2*i] = 1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        
        # Wall 1: 0.5 - x - t
        jac[wall_idx, 2*i] = -1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        
        # Wall 2: y + 0.5 - t
        jac[wall_idx, 2*i+1] = 1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        
        # Wall 3: 0.5 - y - t
        jac[wall_idx, 2*i+1] = -1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
    
    return jac


def _slsqp_objective_grad(x):
    """
    Objective and gradient for SLSQP.
    Objective: -t, gradient: [0, 0, ..., 0, -1]
    """
    t = x[2*N]
    obj = -t
    grad = np.zeros(2*N + 1)
    grad[2*N] = -1.0
    return obj, grad


def _grid_init(rows, cols):
    """Generate a grid initial layout."""
    # Place N circles in a grid of rows x cols
    # We need to fit N into rows*cols.
    # For N=50, 7x7=49 is too small. 7x8=56.
    # We will use a 7x8 grid and remove 6 points, or just use 8x7?
    # Better: Just use a grid that covers the square.
    
    # Use aspect ratio to determine rows/cols
    # For square, rows ~= cols
    # 7x8 grid:
    x = np.linspace(-HALF, HALF, cols)
    y = np.linspace(-HALF, HALF, rows)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # If we have more points than N, subsample
    if len(grid) > N:
        # Remove points from the edge or random?
        # Just take the first N? Or center them?
        # Centering is better.
        # We generated a full grid. Let's just take a subset.
        # To be fair, let's re-generate exactly N points in a grid-like manner.
        pass
        
    # Simple robust init:
    # Map indices to grid positions
    pts = []
    # Try to find best R, C such that R*C >= N
    best_r, best_c = 7, 8
    for r in range(5, 10):
        for c in range(5, 10):
            if r * c >= N:
                # Prefer square-ish
                if abs(r-c) < abs(best_r-best_c):
                    best_r, best_c = r, c
    
    x = np.linspace(-HALF, HALF, best_c)
    y = np.linspace(-HALF, HALF, best_r)
    
    # Create full grid
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # Shuffle and take N? No, that breaks grid structure.
    # Remove points from the corners/edges to maintain symmetry?
    # Or just take the first N in a space-filling curve?
    
    # Simplest: Take the first N points of the grid (top-left to bottom-right)
    # This might bias the packing.
    # Better: Randomly select N points from the grid?
    # Or, generate a grid of exactly N points by adjusting spacing?
    
    # Let's just use the 7x8 grid and drop 6 points randomly from the perimeter
    # to keep the core intact.
    idx = np.arange(len(full_grid))
    # Prioritize keeping center
    # Distance from center
    dists_to_center = np.sum(full_grid**2, axis=1)
    # Sort by distance, keep closest N
    sorted_idx = np.argsort(dists_to_center)
    pts = full_grid[sorted_idx[:N]]
    return pts


def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    # Hexagonal close packing
    # Row spacing: r * sqrt(3)
    # Col spacing: 2r
    # Let's use a fixed r for init, say 0.07
    r_init = 0.07
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    # Determine grid size
    max_x = HALF - r_init
    max_y = HALF - r_init
    
    row = 0
    y = -max_y
    while y <= max_y:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 2: # Safety break
            break
            
    pts = np.array(points)
    # Shuffle and take N?
    # Or take closest to center?
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts


def _random_init(rng):
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3):
    """
    Multi-stage optimization using epigraph formulation.
    Returns optimized centers.
    """
    c0 = x0.reshape(N, 2)
    r0 = _score_from_centres(c0)
    t0_init = max(r0, 0.01)
    
    x = np.concatenate([c0.ravel(), [t0_init]])
    
    bounds_c = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        try:
            res = minimize(
                lambda x_: _epigraph_objective(x_, beta=beta),
                x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
            )
            x = res.x
        except Exception:
            pass
    
    c = x[:2*N].reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    return c


def _optimize_with_slsqp(x0, max_iter=500, ftol=1e-14):
    """
    Optimize using SLSQP with hard constraints.
    x = [centers, t], maximize t subject to all constraints.
    """
    c0 = x0.reshape(N, 2)
    r0 = _score_from_centres(c0)
    t0_init = max(r0 * 0.999, 0.001)  # Start slightly below current
    
    x = np.concatenate([c0.ravel(), [t0_init]])
    
    bounds_c = [(-HALF + 1e-12, HALF - 1e-12)] * (2 * N)
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    try:
        res = minimize(
            lambda x_: _slsqp_objective(x_),
            x,
            method='SLSQP',
            bounds=bounds,
            jac=lambda x_: _slsqp_objective_grad(x_)[1],
            constraints=[{
                'type': 'ineq',
                'fun': _slsqp_constraint_func,
                'jac': _slsqp_jacobian
            }],
            options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
        )
        
        if res.success or res.fun < -t0_init:
            c_opt = res.x[:2*N].reshape(N, 2)
            c_opt = np.clip(c_opt, -HALF + 1e-12, HALF - 1e-12)
            return c_opt
        else:
            return c0
    
    except Exception:
        return c0


def _coordinate_descent_polish(c, max_iters=1000, tol=1e-15):
    """
    Coordinate descent polish: for each circle, try to move it to increase the
    minimum distance. This directly works with the hard min function.
    """
    c = c.copy()
    c = np.clip(c, -HALF + 1e-12, HALF - 1e-12)
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Get current min distance and which constraint is active
            r_cur, pair_idx, wall_idx = _min_distance_and_indices(c)
            
            # Try small moves in each direction
            best_move = None
            best_r = r_cur
            
            # Try moving circle i in 4 directions and 4 magnitudes
            for dx in [-0.001, -0.0001, 0.0001, 0.001]:
                for dy in [-0.001, -0.0001, 0.0001, 0.001]:
                    c_trial = c.copy()
                    c_trial[i, 0] += dx
                    c_trial[i, 1] += dy
                    c_trial = np.clip(c_trial, -HALF + 1e-12, HALF - 1e-12)
                    
                    r_trial = _score_from_centres(c_trial)
                    if r_trial > best_r + 1e-16:
                        best_r = r_trial
                        best_move = (dx, dy)
            
            if best_move is not None:
                c[i, 0] += best_move[0]
                c[i, 1] += best_move[1]
                c[i] = np.clip(c[i], -HALF + 1e-12, HALF - 1e-12)
                improved = True
        
        if not improved:
            break
    
    return c


def _polish_nm(c, timeout=10.0):
    """
    Final high-precision polish using Nelder-Mead to maximize the minimum distance.
    """
    t_start = time.time()
    
    def objective(x):
        c_local = x.reshape(N, 2)
        c_local = np.clip(c_local, -HALF + 1e-12, HALF - 1e-12)
        d = pdist(c_local)
        min_pair = d.min() if d.size > 0 else 0.0
        
        dist_walls = np.min(np.stack([
            c_local[:, 0] + HALF,
            HALF - c_local[:, 0],
            c_local[:, 1] + HALF,
            HALF - c_local[:, 1]
        ]), axis=0)
        min_wall = dist_walls.min()
        
        r = min(min_pair / 2.0, min_wall)
        return -r

    x0 = c.ravel()
    
    options = {
        'maxiter': 2000,
        'xatol': 1e-14,
        'fatol': 1e-14,
        'adaptive': True,
        'disp': False
    }
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options=options
        )
        c_opt = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        
        s_opt = _score_from_centres(c_opt)
        s_init = _score_from_centres(c)
        
        if s_opt > s_init:
            return c_opt
        else:
            return c
            
    except Exception:
        return c

def _polish_high_precision(c, max_iter=1000):
    """
    Final high-precision polish using very high beta soft-min.
    """
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    # Use progressively higher beta
    for beta in [1000.0, 2000.0, 5000.0]:
        try:
            res = minimize(
                lambda cf: _negative_smooth_score(cf, beta=beta),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16, 'iprint': 0}
            )
            x0 = res.x
        except Exception:
            pass
    
    c = np.clip(x0.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    return c


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial configurations
    inits = [
        _grid_init(7, 8),
        _hex_init(),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    # Optimization loop
    n_starts = 25
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 30:
            break
            
        if i < len(inits):
            x0 = inits[i].ravel()
        else:
            if i % 3 == 0:
                x0 = best_centres.ravel() + rng.normal(0, 0.01, size=2*N)
            elif i % 3 == 1:
                x0 = best_centres.ravel() + rng.normal(0, 0.03, size=2*N)
            else:
                x0 = _random_init(rng).ravel()
                
        x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
        
        try:
            # Step 1: Epigraph optimization (fast, good for global)
            c_opt = _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3)
            
            # Step 2: SLSQP with hard constraints (more precise)
            if time.time() - t0 < DEADLINE - 25:
                c_slsqp = _optimize_with_slsqp(c_opt, max_iter=300, ftol=1e-13)
                s_slsqp = _score_from_centres(c_slsqp)
                s_epigraph = _score_from_centres(c_opt)
                if s_slsqp > s_epigraph:
                    c_opt = c_slsqp
            
            # Step 3: High-precision polish
            if time.time() - t0 < DEADLINE - 15:
                c_opt = _polish_high_precision(c_opt, max_iter=300)
            
            s = _score_from_centres(c_opt)
            if s > best_score:
                best_score = s
                best_centres = c_opt
                
        except Exception:
            pass

    # Final polish on best result
    if time.time() - t0 < DEADLINE - 10:
        try:
            # SLSQP final
            best_centres = _optimize_with_slsqp(best_centres, max_iter=500, ftol=1e-14)
            s1 = _score_from_centres(best_centres)
            if s1 > best_score:
                best_score = s1
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 8:
        try:
            # High-precision L-BFGS
            best_centres = _polish_high_precision(best_centres, max_iter=500)
            s2 = _score_from_centres(best_centres)
            if s2 > best_score:
                best_score = s2
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 5:
        try:
            # Coordinate descent polish
            best_centres = _coordinate_descent_polish(best_centres, max_iters=200)
            s3 = _score_from_centres(best_centres)
            if s3 > best_score:
                best_score = s3
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 3:
        try:
            # Final SLSQP
            best_centres = _optimize_with_slsqp(best_centres, max_iter=200, ftol=1e-15)
            s4 = _score_from_centres(best_centres)
            if s4 > best_score:
                best_score = s4
        except Exception:
            pass

    # Final validation
    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    return best_centres
