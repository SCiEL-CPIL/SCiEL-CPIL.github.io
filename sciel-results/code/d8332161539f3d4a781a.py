"""
Optimization of 100 equal spheres in a unit cube.
Uses a lattice-based greedy initialization followed by L-BFGS-B optimization 
and a final high-precision polish pass.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    # dist to wall = 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_radius(centers, beta=50.0):
    """
    Computes a smooth surrogate for the minimum radius using log-sum-exp.
    r_soft = -1/beta * log(sum(exp(-beta * r_i)))
    where r_i are the individual constraint distances (pairwise/2 and wall).
    """
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    # We only need upper triangle to avoid double counting, but for log-sum-exp 
    # summing all pairs is fine if we are consistent. 
    # Actually, standard soft min over a set of values.
    # Values: d_ij / 2 for i<j, and w_i for walls.
    
    # To be efficient and accurate, let's construct the array of constraint values
    # N(N-1)/2 pairs + N walls.
    # For N=100, pairs = 4950. Total ~5050 values.
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraint distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow/underflow, subtract max
    max_d = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_d))
    log_sum = np.log(np.sum(exp_terms))
    
    r_soft = max_d - log_sum / beta
    return r_soft

def _energy_and_gradient(x, target_R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    We use a smooth approximation for the distance to handle non-differentiability at d=0.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # We want to minimize sum((2R - d)^2) for d < 2R
    # Gradient of (2R - d)^2 w.r.t center_i:
    # d/dx_i (2R - d)^2 = 2(2R - d) * (-1) * d(-d)/dx_i
    # d(-d)/dx_i = (x_i - x_j) / d
    # So term is -2(2R - d) * (x_i - x_j) / d
    # Note: d is distance.
    
    mask = dists < 2.0 * target_R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        m_ij = mask[i_idx, j_idx]
        
        if np.any(m_ij):
            v_ij = 2.0 * target_R - d_ij[m_ij]
            energy += np.sum(v_ij**2)
            
            # Vectorized gradient update
            # Force on i from j: -2 * v * (c_i - c_j) / d
            # We need to accumulate these into grad
            # Let's use a loop for clarity as N is small, or vectorize with scatter.
            # Vectorized scatter is tricky with variable counts. Loop is fine for N=100.
            for k in range(len(m_ij)):
                if not m_ij[k]:
                    continue
                i = i_idx[k]
                j = j_idx[k]
                v = v_ij[k]
                d = d_ij[k]
                # factor = -2 * v / d
                factor = -2.0 * v / d
                vec = centers[i] - centers[j]
                force = factor * vec
                grad[i] += force
                grad[j] -= force

    # 2. Wall terms
    # Constraint: 0.5 - max(|x|, |y|, |z|) >= R
    # Violation: R - (0.5 - max_abs) = R - 0.5 + max_abs
    # Let d_wall = 0.5 - max_abs. Violation v = R - d_wall.
    # Energy += v^2.
    # Grad: d(v^2)/dc = 2v * dv/dc.
    # v = R - 0.5 + max_abs.
    # dv/dc = d(max_abs)/dc.
    # d(max(|x|,|y|,|z|))/dx = sign(x) if |x| is max, else 0.
    
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < target_R
    if np.any(mask_wall):
        v_wall = (target_R - wall_dists)[mask_wall]
        energy += np.sum(v_wall**2)
        
        indices_wall = np.where(mask_wall)[0]
        for i in indices_wall:
            v = target_R - wall_dists[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            # Find which coordinate is max
            # If multiple are max, gradient is sub-gradient. We pick the first one or average.
            # Picking first is stable enough for optimization.
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            # dv/dc = grad_wall_i
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_budget=5.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
        )
        
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    Final high-precision polish.
    Maximizes the minimum radius directly.
    We minimize -soft_min_radius(centers) with a high beta to approximate exact min.
    """
    start_time = time.time()
    
    def neg_soft_min(x):
        c = x.reshape(-1, 3)
        return -_soft_min_radius(c, beta=100.0)
    
    # L-BFGS-B requires gradient. We can use numerical gradient for this small polish step
    # or implement analytical gradient of soft min.
    # Given time constraints, numerical gradient with L-BFGS-B is risky if not careful.
    # However, N=100, 300 vars. NumPy gradient is fast.
    
    # Alternative: Coordinate ascent on the "critical" constraints.
    # Let's stick to a robust L-BFGS-B with numerical gradient if analytical is too complex to derive correctly for soft-min.
    # Actually, let's just do a few steps of L-BFGS-B on the original energy function with R = current_best_radius + epsilon.
    # Or simpler: use the existing _optimize_for_radius with R slightly higher than current best.
    
    # For this polish, we'll use a tight L-BFGS-B on the energy function for R = current_r * 1.001
    r_curr = _compute_radius(centers)
    target_R = r_curr * 1.001 + 1e-6
    
    # If we are very close to the limit, just return
    if target_R > 0.115: # Safety cap, known max is ~0.1114
        target_R = 0.1114
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return centers

def _greedy_mis_init(target_dist):
    """
    Initializes N centers using a greedy maximal independent set on a dense lattice.
    """
    # Lattice size 20x20x20 = 8000 points.
    # Spacing 1/20 = 0.05.
    # Target dist ~ 0.22 (for r=0.11).
    # 0.22 / 0.05 = 4.4 lattice units.
    
    L = 20
    coords = np.linspace(-0.5, 0.5, L)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    
    # Shuffle to make it random-ish, but deterministic seed
    rng = np.random.RandomState(42)
    rng.shuffle(grid)
    
    selected = []
    for p in grid:
        if len(selected) == N:
            break
        # Check distance to all selected
        if not selected:
            selected.append(p)
        else:
            dists = np.linalg.norm(np.array(selected) - p, axis=1)
            if np.min(dists) >= target_dist:
                selected.append(p)
                
    # If we didn't get N, fill with random points that are as far as possible
    while len(selected) < N:
        # Pick a random point from grid not yet selected
        remaining = [g for g in grid if g not in selected]
        if not remaining:
             # Fallback to random
             p = rng.uniform(-0.5, 0.5, size=3)
        else:
             # Pick the one that maximizes min distance to selected
             best_p = None
             best_min_d = -1
             # Sample a subset to speed up
             sample = rng.choice(len(remaining), size=min(100, len(remaining)), replace=False)
             for idx in sample:
                 p = remaining[idx]
                 dists = np.linalg.norm(np.array(selected) - p, axis=1)
                 min_d = np.min(dists)
                 if min_d > best_min_d:
                     best_min_d = min_d
                     best_p = p
             selected.append(best_p)
             
    return np.array(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_radius = 0.0
    best_centers = None
    
    # Strategy: Greedy MIS on lattice
    # Start with a target distance slightly larger than 2 * 0.11138
    # 2 * 0.11138 = 0.22276
    # Let's try 0.220 to ensure we can find 100 points.
    init_centers = _greedy_mis_init(0.220)
    
    r_est = _compute_radius(init_centers)
    
    if r_est > best_radius:
        best_radius = r_est
        best_centers = init_centers.copy()
        
    # Iterative refinement
    # We will try to increase the radius from r_est
    target_R = r_est * 1.01
    if target_R < 0.05:
        target_R = 0.05
        
    current_centers = init_centers.copy()
    
    # Main loop
    while time.time() < deadline - 15.0:
        # Optimize for target_R
        time_budget = min(10.0, max(2.0, deadline - 5.0 - time.time()))
        optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=100, time_budget=time_budget)
        
        # Check result
        r_achieved = _compute_radius(optimized_centers)
        
        if r_achieved > best_radius:
            best_radius = r_achieved
            best_centers = optimized_centers.copy()
            
        # If we successfully fit target_R (energy is small), increase it
        if energy < 1e-8:
            # Increase target
            if target_R < 0.1113:
                target_R *= 1.002
            else:
                target_R += 1e-5
            current_centers = optimized_centers
        else:
            # Failed to fit. 
            # Check if we improved the current best with this attempt anyway
            if r_achieved > best_radius:
                 best_radius = r_achieved
                 best_centers = optimized_centers.copy()
            
            # If we are stuck, try to relax target_R slightly to recover, then increase again
            # Or just break if we are very close to the limit
            if target_R > 0.111:
                # Try to polish the best found so far
                break
                
            # Relax target
            target_R *= 0.99
            current_centers = optimized_centers # Use the partially optimized centers
            
    # Final Polish
    if time.time() < deadline - 5.0 and best_centers is not None:
        polished = _polish_centers(best_centers, max_iter=100, time_budget=10.0)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
            
    # Safety check
    if best_centers is None:
        # Fallback to grid
        xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
        ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 4)
        zs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        best_centers = grid[:N]
    else:
        # Ensure centers are strictly inside
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        
    return best_centers

# EVOLVE-BLOCK-END
