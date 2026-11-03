"""
Circle packing in a unit square: maximize common radius for 50 equal circles.
Strategy: Direct constrained optimization using SLSQP (Interior Point method).
We maximize r subject to ||c_i - c_j||^2 >= 4r^2 and wall constraints.
This directly optimizes the active set (contacts) rather than using a smooth surrogate.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _init_hex():
    """
    Generate a hexagonal lattice initialization.
    This is a good starting point for circle packing as it is locally dense.
    We scale it to fit inside the square.
    """
    positions = []
    # Use a spacing that allows for 50 circles roughly
    # Hex packing density is ~0.9069. Square area 1. 50 circles area 50*pi*r^2.
    # r ~ 0.07 => area ~ 0.015 => total area ~ 0.77.
    # Spacing d = 2r ~ 0.14.
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    
    # Generate a grid larger than needed, then crop
    # To fit 50 circles in a square of side 1, we need roughly 7x7 or 8x7 points
    # Let's generate a 10x10 grid and pick the best 50 or just scale a 7x7.
    # Actually, a 7x7 grid has 49. An 8x8 has 64.
    # Let's try a 7x7 grid centered, plus one in the middle if needed?
    # No, let's just generate a dense hex grid and select the 50 that fit best or just scale.
    
    # Simpler: Generate a 7x7 hex grid, scale to fit [-0.5, 0.5].
    # 7x7 = 49 circles. We need 50.
    # Let's use an 8x8 grid, take the first 50, and scale.
    
    rows = 8
    cols = 8
    for row in range(rows):
        for col in range(cols):
            x = col * spacing + (0.5 * spacing if row % 2 else 0)
            y = row * row_spacing
            positions.append([x, y])
            
    positions = np.array(positions[:N])
    
    # Center and scale
    # Find bounding box
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    # Shift to center
    positions[:, 0] -= (min_x + max_x) / 2
    positions[:, 1] -= (min_y + max_y) / 2
    
    # Scale to fit within [-0.45, 0.45] to leave some room for optimization
    max_abs = np.abs(positions).max()
    if max_abs > 0:
        positions *= (0.45 / max_abs)
        
    return positions

def _init_grid():
    """
    Simple grid initialization.
    """
    # 7x7 grid + 1 center?
    # Let's do an 8x8 grid, pick 50.
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _init_rand(rng):
    """Random initialization."""
    return rng.uniform(-0.4, 0.4, (N, 2))

def _slsqp_optimize(x0, t0, deadline, maxiter=500, ftol=1e-12):
    """
    Optimize using SLSQP.
    Variables: x = [c_0, c_1, ..., c_{N-1}, r]
    Objective: Minimize -r
    Constraints:
    1. Wall constraints: 0.5 - |c_i| - r >= 0 for all i, both axes.
       This is non-smooth due to |c_i|. 
       To handle this in SLSQP, we can split into 4 constraints per circle:
       c_i_x - r <= 0.5
       -c_i_x - r <= 0.5
       c_i_y - r <= 0.5
       -c_i_y - r <= 0.5
    2. Pairwise constraints: ||c_i - c_j||^2 - 4r^2 >= 0.
       These are smooth quadratic constraints.
    """
    n_vars = 2 * N + 1
    r_idx = 2 * N
    
    def objective(x):
        return -x[r_idx]

    def objective_grad(x):
        g = np.zeros(n_vars)
        g[r_idx] = -1.0
        return g

    # Constraints
    # We will define them as h(x) >= 0 for SLSQP (which expects 'ineq' as >= 0)
    
    # Wall constraints (4 per circle)
    # c_i_x - r - 0.5 <= 0  => 0.5 + r - c_i_x >= 0
    # -c_i_x - r - 0.5 <= 0 => -0.5 + r + c_i_x >= 0
    # c_i_y - r - 0.5 <= 0  => 0.5 + r - c_i_y >= 0
    # -c_i_y - r - 0.5 <= 0 => -0.5 + r + c_i_y >= 0
    
    def wall_constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        cons = np.empty(4 * N)
        for i in range(N):
            cons[4*i]   = 0.5 + r - c[i, 0]
            cons[4*i+1] = -0.5 + r + c[i, 0]
            cons[4*i+2] = 0.5 + r - c[i, 1]
            cons[4*i+3] = -0.5 + r + c[i, 1]
        return cons

    def wall_constraints_grad(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        J = np.zeros((4 * N, n_vars))
        for i in range(N):
            # d(0.5 + r - c_i_x)/dx
            J[4*i, 2*i] = -1.0
            J[4*i, r_idx] = 1.0
            
            # d(-0.5 + r + c_i_x)/dx
            J[4*i+1, 2*i] = 1.0
            J[4*i+1, r_idx] = 1.0
            
            # d(0.5 + r - c_i_y)/dx
            J[4*i+2, 2*i+1] = -1.0
            J[4*i+2, r_idx] = 1.0
            
            # d(-0.5 + r + c_i_y)/dx
            J[4*i+3, 2*i+1] = 1.0
            J[4*i+3, r_idx] = 1.0
        return J

    # Pairwise constraints: ||c_i - c_j||^2 - 4r^2 >= 0
    # Number of pairs: N*(N-1)/2
    n_pairs = N * (N - 1) // 2
    
    def pair_constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        cons = np.empty(n_pairs)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                cons[idx] = np.dot(diff, diff) - 4 * r * r
                idx += 1
        return cons

    def pair_constraints_grad(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        J = np.zeros((n_pairs, n_vars))
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                # d(||c_i - c_j||^2 - 4r^2)/d(c_i) = 2 * diff
                J[idx, 2*i] = 2 * diff[0]
                J[idx, 2*i+1] = 2 * diff[1]
                # d(...)/d(c_j) = -2 * diff
                J[idx, 2*j] = -2 * diff[0]
                J[idx, 2*j+1] = -2 * diff[1]
                # d(...)/dr = -8r
                J[idx, r_idx] = -8 * r
                idx += 1
        return J

    constraints = [
        {'type': 'ineq', 'fun': wall_constraints, 'jac': wall_constraints_grad},
        {'type': 'ineq', 'fun': pair_constraints, 'jac': pair_constraints_grad}
    ]
    
    # Bounds for centers: [-0.5, 0.5]
    # Bounds for r: [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            jac=objective_grad, 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        return result.x, result.success
    except Exception as e:
        # If SLSQP fails, return the input
        return x0, False

def _slsqp_polish(centers, r, t0, deadline, maxiter=500):
    """
    Run SLSQP polish on a given configuration.
    """
    x0 = np.zeros(2 * N + 1)
    x0[:2*N] = centers.flatten()
    x0[2*N] = r
    
    x_res, success = _slsqp_optimize(x0, t0, deadline, maxiter=maxiter)
    
    if success:
        c_res = x_res[:2*N].reshape(N, 2)
        r_res = x_res[2*N]
        # Verify validity
        r_check = _score(c_res)
        if r_check >= 0:
            return c_res, r_check
    return centers, r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initializations
    # 1. Hexagonal
    c_hex = _init_hex()
    r_hex = _score(c_hex)
    
    # 2. Grid
    c_grid = _init_grid()
    r_grid = _score(c_grid)
    
    # 3. Random
    c_rand = _init_rand(rng)
    r_rand = _score(c_rand)
    
    best_c = c_hex
    best_r = r_hex
    if r_grid > best_r:
        best_c = c_grid
        best_r = r_grid
    if r_rand > best_r:
        best_c = c_rand
        best_r = r_rand
        
    # Strategy:
    # 1. Run SLSQP on the best init.
    # 2. If time permits, run SLSQP on other inits or restart from the result of the first.
    # 3. SLSQP is local, so multiple starts are beneficial.
    
    # Phase 1: Polish the best initial guess
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_new, r_new = _slsqp_polish(best_c, best_r, t0, DEADLINE, maxiter=200)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
            
    # Phase 2: Try other inits if time permits
    inits_to_try = []
    if r_hex > r_grid and r_hex > r_rand:
        inits_to_try = [(c_grid, r_grid), (c_rand, r_rand)]
    elif r_grid > r_hex and r_grid > r_rand:
        inits_to_try = [(c_hex, r_hex), (c_rand, r_rand)]
    else:
        inits_to_try = [(c_hex, r_hex), (c_grid, r_grid)]
        
    for c_init, r_init in inits_to_try:
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        # Only try if this init is significantly worse than best, or if we have lots of time
        # Actually, let's just try them to escape local minima.
        c_new, r_new = _slsqp_polish(c_init, r_init, t0, DEADLINE, maxiter=200)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
            
    # Phase 3: Restart from the current best with a slight perturbation to escape local minima
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        # Perturb
        c_perturbed = best_c + rng.normal(0, 0.005, best_c.shape)
        c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
        r_perturbed = _score(c_perturbed)
        if r_perturbed > 0:
            c_new, r_new = _slsqp_polish(c_perturbed, r_perturbed, t0, DEADLINE, maxiter=300)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                
    # Phase 4: Final polish with higher maxiter if time allows
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_new, r_new = _slsqp_polish(best_c, best_r, t0, DEADLINE, maxiter=500)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new

    # Final check
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        best_c = c_hex
        
    return best_c
