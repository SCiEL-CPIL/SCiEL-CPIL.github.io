import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(pos):
    """Compute the maximum radius for a given set of positions."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pos)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist, min_wall_dist) / 2.0

def _energy_and_grad(pos_flat, target_r):
    """
    Compute energy and gradient for a target radius approach.
    Energy is sum of squared violations of pairwise and wall constraints.
    """
    pos = pos_flat.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(pos_flat)
    
    # Pairwise constraints: dist(i,j) >= 2*target_r
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0.0, 2.0 * target_r - d_ij)
    
    energy += np.sum(v_ij ** 2)
    
    mask = v_ij > 0
    if np.any(mask):
        i_v = i_idx[mask]
        j_v = j_idx[mask]
        v_v = v_ij[mask]
        d_v = d_ij[mask]
        term = 2.0 * v_v / d_v
        grad[i_v] += -term[:, np.newaxis] * diff[i_v]
        grad[j_v] += term[:, np.newaxis] * diff[i_v]
    
    # Wall constraints: 0.5 - |pos_ik| >= target_r
    bound = 0.5 - target_r
    v_pos = np.maximum(0.0, pos - bound)
    energy += np.sum(v_pos ** 2)
    grad += 2.0 * v_pos
    v_neg = np.maximum(0.0, -bound - pos)
    energy += np.sum(v_neg ** 2)
    grad += -2.0 * v_neg
    
    return energy, grad

def _optimize_for_radius(pos0, r_target, max_iter=300):
    """
    Try to find a configuration with radius at least r_target starting from pos0.
    Returns (success, pos).
    """
    pos0_flat = pos0.flatten()
    
    def objective(x):
        e, _ = _energy_and_grad(x, r_target)
        return e
    
    def gradient(x):
        _, g = _energy_and_grad(x, r_target)
        return g
        
    try:
        res = minimize(objective, pos0_flat, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        final_energy = res.fun
        if final_energy < 1e-10:
            pos_final = res.x.reshape(N, 3)
            pos_final = np.clip(pos_final, -0.5, 0.5)
            return True, pos_final
        else:
            return False, pos0
    except Exception:
        return False, pos0

def _soft_min_energy(pos_flat, temp):
    """
    Soft-min objective: maximize the minimum distance by minimizing a soft-min of distances.
    E = -log(sum(exp(-d_i/temp))) / log(1) ... actually just -logsumexp(-d/temp)
    We want to maximize min(d), so minimize -min(d).
    Soft approximation: -logsumexp(-d/temp) where d are the constrained distances.
    """
    pos = pos_flat.reshape(N, 3)
    # Constrained distances: pairwise and wall
    # For pairwise: d_ij = dist(i,j) / 2
    # For wall: d_ik = 0.5 - |pos_ik|
    
    # This is expensive to compute exactly with logsumexp over all pairs.
    # Instead, use a simpler penalty: sum of (target - d)^2 for d < target is the hard approach.
    # For soft approach, let's use: sum( exp(-(d_i - r_current) / temp) ) for d_i < r_current + margin
    # Actually, the standard "soft-min" for packing is to minimize:
    # sum_{i<j} exp(-dist(i,j)/T) + sum_i exp(-wall_dist_i/T)
    # Minimizing this pushes all distances up.
    
    # Let's stick to the target-radius approach as it's more direct for improvement.
    # But we need a better initialization and restart strategy.
    pass

def _initialize_grid():
    """Create a grid-like initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    return pos

def _initialize_random():
    """Create a random initial configuration."""
    pos = np.random.uniform(-0.45, 0.45, (N, 3))
    return pos

def _initialize_lattice():
    """
    Try to create a body-centered cubic (BCC) or face-centered cubic (FCC) like lattice
    restricted to the cube.
    """
    # For N=100, a simple grid is decent. Let's try to perturb a grid.
    pos = _initialize_grid()
    pos += np.random.normal(0, 0.02, (N, 3))
    pos = np.clip(pos, -0.49, 0.49)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Strategy:
    # 1. Generate multiple diverse initializations.
    # 2. For each, run a multi-stage optimization:
    #    a. Coarse search: increase radius in large steps.
    #    b. Fine search: increase radius in small steps.
    #    c. Polish: final squeeze.
    # 3. Keep the best result found.
    
    best_pos = None
    best_r = 0.0
    
    # List of initializers
    initializers = [
        _initialize_grid,
        _initialize_lattice,
        _initialize_random,
    ]
    
    # We will try each initializer, and for each, run the optimization loop.
    # To save time, we might limit the number of restarts or iterations per restart.
    
    num_restarts = 5  # Total number of different starting points to try
    
    for restart_idx in range(num_restarts):
        if time.time() > deadline - 10:  # Need at least 10s left for a full attempt
            break
            
        # Choose initializer
        if restart_idx < len(initializers):
            pos = initializers[restart_idx]()
        else:
            # Additional random restarts
            pos = _initialize_random()
        
        # Add noise to break symmetry for grid/lattice
        if restart_idx < 2:
            pos += np.random.normal(0, 0.01, (N, 3))
            pos = np.clip(pos, -0.49, 0.49)
        
        current_r = _compute_radius(pos)
        
        # Multi-stage optimization
        # Stage 1: Coarse increase
        r = current_r
        step = 0.005
        max_coarse_steps = 20
        
        for _ in range(max_coarse_steps):
            if time.time() > deadline - 10:
                break
            r_new = r + step
            success, pos_new = _optimize_for_radius(pos, r_new, max_iter=200)
            if success:
                actual_r = _compute_radius(pos_new)
                if actual_r > r:
                    r = actual_r
                    pos = pos_new
                else:
                    step *= 0.5
                    if step < 1e-5:
                        break
            else:
                step *= 0.5
                if step < 1e-5:
                    break
        
        # Stage 2: Fine increase
        step = 0.0005
        max_fine_steps = 50
        for _ in range(max_fine_steps):
            if time.time() > deadline - 10:
                break
            r_new = r + step
            success, pos_new = _optimize_for_radius(pos, r_new, max_iter=300)
            if success:
                actual_r = _compute_radius(pos_new)
                if actual_r > r:
                    r = actual_r
                    pos = pos_new
                else:
                    step *= 0.5
                    if step < 1e-7:
                        break
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        
        # Stage 3: Polish (very small steps)
        step = 1e-5
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            r_new = r + step
            success, pos_new = _optimize_for_radius(pos, r_new, max_iter=500)
            if success:
                actual_r = _compute_radius(pos_new)
                if actual_r > r:
                    r = actual_r
                    pos = pos_new
                else:
                    break
            else:
                break
        
        # Update best
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # If we hit a high score, we might stop early or continue to try to improve
        # But given time constraints, we should try a few more restarts if time permits.
    
    if best_pos is None:
        # Fallback: return a valid grid configuration
        best_pos = _initialize_grid()
    
    return best_pos
