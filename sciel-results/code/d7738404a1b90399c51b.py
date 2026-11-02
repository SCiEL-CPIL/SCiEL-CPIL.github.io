"""
Optimized packing of 50 equal circles in a unit square.
Uses Simulated Annealing for global exploration and SLSQP for precise local refinement.
"""
import numpy as np
import time
import scipy.optimize as opt
import sys

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    Returns 0.0 if invalid.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5) or \
       np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return 0.0

    # Wall distances: min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Equivalent to 0.5 - |x|, 0.5 - |y|
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_violation(centers, r_target, beta):
    """
    Compute sum of soft violations for a given radius.
    Lower is better. 0 means feasible.
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = np.maximum(0, r_target - walls)
    
    # Pair violations
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = np.maximum(0, 2.0 * r_target - dists)
    
    # Soft min approximation: sum(exp(-beta * viol)) or similar?
    # Actually, for SA, we just want a scalar cost.
    # Sum of violations is good, but can be flat.
    # Sum of squared violations is better for gradient-like steps, but we are doing SA.
    # Let's use sum of violations + small penalty for large violations to guide SA.
    cost = np.sum(wall_viol) + np.sum(pair_viol)
    
    # Add a quadratic penalty to ensure we push hard against constraints when violated
    # This helps SA escape "soft" local minima where many small violations exist
    cost += 0.1 * (np.sum(wall_viol**2) + np.sum(pair_viol**2))
    
    return float(cost)

def _hex_init(rng):
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    for attempt in range(10):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)
        
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()
            
    return best_grid

def _grid_init(rng):
    """Create a grid layout with jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    grid = grid + rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _simulated_annealing(initial_pts, r_target, time_budget, rng):
    """
    Run Simulated Annealing to minimize violation for a fixed r_target.
    Returns the best points found.
    """
    start_time = time.time()
    current_pts = initial_pts.copy()
    current_cost = _soft_min_violation(current_pts, r_target, 1.0)
    
    best_pts = current_pts.copy()
    best_cost = current_cost
    
    T = 0.1
    T_min = 1e-6
    alpha = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
            
        # Propose a move: pick a random point and move it randomly
        idx = rng.integers(0, N)
        # Step size scales with temperature
        step_size = T * 0.5
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_pts = current_pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        
        # Keep within bounds
        new_pts[idx, 0] = np.clip(new_pts[idx, 0], -0.5, 0.5)
        new_pts[idx, 1] = np.clip(new_pts[idx, 1], -0.5, 0.5)
        
        new_cost = _soft_min_violation(new_pts, r_target, 1.0)
        
        # Acceptance probability
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            current_pts = new_pts
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts.copy()
        
        T *= alpha
        
    return best_pts

def _slsqp_max_radius(initial_pts, r_target, time_budget):
    """
    Use SLSQP to maximize radius starting from initial_pts.
    We maximize r subject to:
    1. dist(i,j) >= 2*r
    2. 0.5 - |x_i| >= r
    3. 0.5 - |y_i| >= r
    """
    start_time = time.time()
    
    # Variables: x_1..x_N, y_1..y_N, r
    # Total 2N + 1 variables
    x0 = np.concatenate([initial_pts.flatten(), [r_target]])
    
    def objective(x):
        # We want to maximize r, so minimize -r
        return -x[-1]
        
    def constraints(x):
        pts = x[:-1].reshape(N, 2)
        r = x[-1]
        
        cons = []
        
        # Boundary constraints: 0.5 - |x_i| >= r  =>  0.5 - |x_i| - r >= 0
        # This is non-smooth due to abs. SLSQP might struggle.
        # We can split into 4 constraints per point? Or use smooth approximation?
        # Actually, SLSQP can handle non-smooth to some extent, but it's risky.
        # Let's use the fact that we are in [-0.5, 0.5].
        # Constraint: x_i + r <= 0.5
        # Constraint: -x_i + r <= 0.5  => x_i - r >= -0.5
        # Constraint: y_i + r <= 0.5
        # Constraint: -y_i + r <= 0.5  => y_i - r >= -0.5
        
        # Inequality form for SLSQP: fun(x) >= 0
        # 0.5 - x_i - r >= 0
        # 0.5 + x_i - r >= 0
        # 0.5 - y_i - r >= 0
        # 0.5 + y_i - r >= 0
        
        cons.append(0.5 - pts[:, 0] - r)
        cons.append(0.5 + pts[:, 0] - r)
        cons.append(0.5 - pts[:, 1] - r)
        cons.append(0.5 + pts[:, 1] - r)
        
        # Pairwise constraints: dist(i,j) >= 2r
        # dist(i,j) - 2r >= 0
        # This is non-smooth at dist=0, but we assume initial points are distinct.
        ci = pts[_IU[0]]
        cj = pts[_IU[1]]
        dists = np.sqrt(np.sum((ci - cj)**2, axis=1))
        cons.append(dists - 2*r)
        
        return cons
        
    # Flatten constraints for SLSQP
    def flat_constraints(x):
        cons_list = constraints(x)
        return np.concatenate([c.flatten() for c in cons_list])
        
    # Bounds for variables
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.01, 0.1)]
    
    try:
        result = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=flat_constraints,
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success:
            new_pts = result.x[:-1].reshape(N, 2)
            new_r = result.x[-1]
            # Verify
            true_r = _compute_radius(new_pts)
            if true_r > 0:
                return new_pts, true_r
        
        return initial_pts, _compute_radius(initial_pts)
        
    except Exception as e:
        return initial_pts, _compute_radius(initial_pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # 1. Generate initial configurations
    inits = []
    inits.append(_hex_init(rng))
    inits.append(_grid_init(rng))
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    best_init = None
    best_init_r = 0
    for init in inits:
        r = _compute_radius(init)
        if r > best_init_r:
            best_init_r = r
            best_init = init
            
    best_pts = best_init
    best_radius = best_init_r
    
    # 2. Multi-start SA + SLSQP Pipeline
    # We will try to improve the radius by running SA for a slightly larger target radius,
    # then refining with SLSQP.
    
    # Estimate a reasonable upper bound
    r_high = 0.075
    
    # We will iterate over a few candidate target radii
    # Start with the best found radius and try to push it higher
    
    current_target = best_radius
    max_sa_time = 5.0
    max_slsqp_time = 10.0
    
    # Loop for time
    while time.time() < deadline - 15:
        # Increase target radius slightly
        # We use a small step to ensure SA has a chance to find a feasible config
        target_r = current_target * 1.005 + 0.001
        
        if target_r > r_high:
             target_r = r_high
             
        # Run SA to try to find a configuration that fits target_r
        # Use best_pts as start
        sa_pts = _simulated_annealing(best_pts, target_r, max_sa_time, rng)
        
        # Check if SA found something better
        sa_r = _compute_radius(sa_pts)
        if sa_r > best_radius:
            best_radius = sa_r
            best_pts = sa_pts
        
        # Run SLSQP to refine
        # SLSQP maximizes r, so we give it the current best pts and a target slightly above current best
        slsqp_pts, slsqp_r = _slsqp_max_radius(best_pts, target_r, max_slsqp_time)
        
        if slsqp_r > best_radius:
            best_radius = slsqp_r
            best_pts = slsqp_pts
            # Update current target to the new best
            current_target = best_radius
        else:
            # If SLSQP didn't improve, we might be stuck.
            # Try a different initial point for SA? Or just increase target less?
            # For now, we just continue, but we should ensure we don't loop indefinitely on same target
            # If no improvement after a few tries, we might need to break or adjust strategy.
            # Simple heuristic: if target_r is very close to best_radius and no improvement, 
            # it's likely a local optimum.
            if target_r - best_radius < 0.0001:
                 # Try a random restart?
                 # Or just break if we've been trying this for a while.
                 # Let's just break to be safe if we are stuck at the limit.
                 pass
                 
    # 3. Final Polish
    if time.time() < deadline - 5:
        # Try one last SLSQP with a tight tolerance
        final_pts, final_r = _slsqp_max_radius(best_pts, best_radius * 1.01, 5.0)
        if final_r > best_radius:
            best_radius = final_r
            best_pts = final_pts

    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# For testing
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
    print(f"Shape: {pts.shape}")
