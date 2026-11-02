"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    # Check bounds
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
        
    # Check pairwise non-overlap
    # Vectorized distance check for all pairs
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    
    # We only care about upper triangle (i < j)
    triu = np.triu_indices(n, 1)
    overlaps = sum_r[triu] - dist[triu]
    
    if np.any(overlaps > 1e-7):
        return False
        
    return True

def _calculate_penalty(xyz):
    """Calculate total penalty for invalidity (overlaps and boundary violations)."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    penalty = 0.0
    
    # Boundary penalties
    lb_x = np.maximum(0, r - x)
    penalty += np.sum(lb_x ** 2)
    ub_x = np.maximum(0, x + r - 1)
    penalty += np.sum(ub_x ** 2)
    lb_y = np.maximum(0, r - y)
    penalty += np.sum(lb_y ** 2)
    ub_y = np.maximum(0, y + r - 1)
    penalty += np.sum(ub_y ** 2)
    
    # Overlap penalties
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    
    triu = np.triu_indices(n, 1)
    overlaps = np.maximum(0, sum_r[triu] - dist[triu])
    penalty += np.sum(overlaps ** 2)
    
    return penalty

def _score(xyz):
    """
    Objective function for Simulated Annealing.
    We want to maximize this score.
    Score = Sum(r) - Weight * Penalty
    If Penalty is 0, Score = Sum(r).
    If Penalty > 0, Score is reduced.
    """
    penalty = _calculate_penalty(xyz)
    sum_r = np.sum(xyz[:, 2])
    # Weight should be high enough to strongly penalize invalid states
    # but not so high that it dominates the radius maximization initially.
    # We adapt this weight or keep it fixed. Fixed is simpler.
    weight = 10.0 
    return sum_r - weight * penalty

def _initial_grid():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Radius slightly less than half cell size to ensure no initial overlap
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_hex():
    """Create a hexagonal initial configuration."""
    positions = []
    # 5x5 grid with offsets
    side = 5
    for j in range(side):
        for i in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.5 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            positions.append([x, y, r])
    # Add 26th circle in center gap or random
    positions.append([0.5, 0.5, 0.1])
    return np.array(positions[:N])

def _initial_random(seed):
    """Create a random initial configuration."""
    np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.12, N)
    # Clip to ensure inside bounds initially (roughly)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    return np.column_stack([x, y, r])

def _simulated_annealing(xyz_init, time_budget):
    """
    Run Simulated Annealing on the given initial configuration.
    Returns the best valid configuration found.
    """
    start_time = time.time()
    xyz = xyz_init.copy()
    
    # Parameters
    T = 1.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    min_T = 1e-5
    max_steps = 100000
    
    best_xyz = xyz.copy()
    best_score = _score(xyz)
    best_valid_score = -np.inf
    
    # Adaptive step sizes
    sigma_pos = 0.05
    sigma_r = 0.02
    
    n_accepted = 0
    n_rejected = 0
    window = 1000
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Generate new state
        xyz_new = xyz.copy()
        
        # Random move type
        move_type = np.random.random()
        
        if move_type < 0.7:
            # Move a random circle's position
            idx = np.random.randint(N)
            dx = np.random.normal(0, sigma_pos)
            dy = np.random.normal(0, sigma_pos)
            xyz_new[idx, 0] += dx
            xyz_new[idx, 1] += dy
            # Clip to [0, 1] roughly to avoid huge penalties
            xyz_new[idx, 0] = np.clip(xyz_new[idx, 0], 0, 1)
            xyz_new[idx, 1] = np.clip(xyz_new[idx, 1], 0, 1)
        elif move_type < 0.9:
            # Change a random circle's radius
            idx = np.random.randint(N)
            dr = np.random.normal(0, sigma_r)
            xyz_new[idx, 2] += dr
            # Ensure radius is positive and not too large
            xyz_new[idx, 2] = np.clip(xxyz_new[idx, 2] if False else xyz_new[idx, 2], 1e-4, 0.5)
            # Also adjust position to stay in bounds if radius increased significantly? 
            # No, let penalty handle it.
        else:
            # Perturb multiple circles slightly (noise)
            indices = np.random.choice(N, size=3, replace=False)
            for idx in indices:
                dx = np.random.normal(0, sigma_pos * 0.5)
                dy = np.random.normal(0, sigma_pos * 0.5)
                dr = np.random.normal(0, sigma_r * 0.5)
                xyz_new[idx, 0] += dx
                xyz_new[idx, 1] += dy
                xyz_new[idx, 2] += dr
                xyz_new[idx, 0] = np.clip(xyz_new[idx, 0], 0, 1)
                xyz_new[idx, 1] = np.clip(xyz_new[idx, 1], 0, 1)
                xyz_new[idx, 2] = np.clip(xyz_new[idx, 2], 1e-4, 0.5)
        
        score_new = _score(xyz_new)
        score_old = _score(xyz)
        
        delta_score = score_new - score_old
        
        if delta_score > 0:
            # Accept better solution
            xyz = xyz_new
            n_accepted += 1
        else:
            # Accept worse solution with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta_score / T)
                if np.random.random() < prob:
                    xyz = xyz_new
                    n_accepted += 1
                else:
                    n_rejected += 1
            else:
                n_rejected += 1
        
        # Track best
        if score_new > best_score:
            best_score = score_new
            best_xyz = xyz_new.copy()
            
        # Track best VALID solution
        # We check validity periodically or every step? 
        # Validity check is O(N^2). N=26 is small. 
        # Let's check every 10 steps to save time, or if score is high.
        if step % 10 == 0 or score_new > best_valid_score + 0.01:
             if _validate(xyz_new):
                 val_score = np.sum(xyz_new[:, 2])
                 if val_score > best_valid_score:
                     best_valid_score = val_score
                     # We don't update best_xyz here to keep the 'best_score' logic consistent,
                     # but we want to return the best valid one.
                     # Let's store it separately.
                     best_valid_xyz = xyz_new.copy()
        
        # Adapt step sizes based on acceptance rate
        if step % window == 0 and step > 0:
            accept_rate = n_accepted / (n_accepted + n_rejected + 1e-9)
            # Target accept rate ~ 0.3-0.5
            if accept_rate > 0.5:
                sigma_pos *= 1.1
                sigma_r *= 1.1
            elif accept_rate < 0.2:
                sigma_pos *= 0.9
                sigma_r *= 0.9
            n_accepted = 0
            n_rejected = 0
        
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T
    
    # Final check on best_xyz
    if _validate(best_xyz):
        return best_xyz
    elif 'best_valid_xyz' in locals():
        return best_valid_xyz
    else:
        # Fallback: shrink best_xyz to make it valid
        xyz_fix = best_xyz.copy()
        for _ in range(100):
            if _validate(xyz_fix):
                return xyz_fix
            xyz_fix[:, 2] *= 0.95
            xyz_fix[:, 0] = np.clip(xyz_fix[:, 0], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
            xyz_fix[:, 1] = np.clip(xyz_fix[:, 1], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
        return _initial_grid()

def _lbfgs_polish(xyz, time_budget):
    """
    Use L-BFGS-B to polish a valid configuration.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    def obj(p):
        n = N
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        
        penalty = 0.0
        lb_x = np.maximum(0, r - x)
        penalty += np.sum(lb_x ** 2)
        ub_x = np.maximum(0, x + r - 1)
        penalty += np.sum(ub_x ** 2)
        lb_y = np.maximum(0, r - y)
        penalty += np.sum(lb_y ** 2)
        ub_y = np.maximum(0, y + r - 1)
        penalty += np.sum(ub_y ** 2)
        neg_r = np.maximum(0, -r)
        penalty += np.sum(neg_r ** 2)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = r[:, None] + r[None, :]
        overlap = np.maximum(0, sum_r - dist)
        triu = np.triu_indices(n, 1)
        penalty += np.sum(overlap[triu] ** 2)
        
        return -np.sum(r) + 100.0 * penalty
    
    bounds = [(0, 1), (0, 1), (1e-4, 0.5)] * N
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        xyz_res = result.x.reshape(N, 3)
        if _validate(xyz_res):
            if np.sum(xyz_res[:, 2]) > np.sum(xyz[:, 2]):
                return xyz_res
    except Exception:
        pass
        
    return xyz

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 145.0  # Safety margin
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # List of initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(456),
        _initial_random(789),
        _initial_random(101),
        _initial_random(2024)
    ]
    
    # Run SA for each initial configuration
    for i, xyz0 in enumerate(inits):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time to each SA run
        # More time for earlier runs? Or equal?
        # Let's do equal allocation for now, but cap it.
        time_budget = min(15.0, remaining - 2.0)
        if time_budget < 2.0:
            break
            
        xyz_opt = _simulated_annealing(xyz0, time_budget)
        score = np.sum(xyz_opt[:, 2])
        
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final Polish with L-BFGS if time permits
    elapsed = time.time() - global_start
    if elapsed < deadline - 5.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 2.0:
            xyz_polished = _lbfgs_polish(best_xyz, remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished
    
    # Final validation
    if not _validate(best_xyz):
        # Try to fix
        xyz_fix = best_xyz.copy()
        for _ in range(300):
            if _validate(xyz_fix):
                break
            xyz_fix[:, 2] *= 0.95
            xyz_fix[:, 0] = np.clip(xyz_fix[:, 0], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
            xyz_fix[:, 1] = np.clip(xyz_fix[:, 1], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
        if _validate(xyz_fix):
            best_xyz = xyz_fix
        else:
            # Fallback to grid
            best_xyz = _initial_grid()
            
    return best_xyz

# EVOLVE-BLOCK-END
