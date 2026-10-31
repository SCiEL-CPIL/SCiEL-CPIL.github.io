"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# Precompute pair indices for vectorized overlap checks
_II, _JJ = np.triu_indices(N, k=1)

def _is_valid(pos, radii, tol=1e-9):
    """
    Strictly check if the configuration is valid.
    pos: (N, 2) array of centers
    radii: (N,) array of radii
    """
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    # Check overlaps
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[_II] + radii[_JJ]
    # We require dist >= r_i + r_j - tol
    # dist_sq >= (r_i + r_j - tol)^2
    # Using a slightly looser check for speed in the loop, strict for final
    if np.any(dist_sq < (min_dist - tol) ** 2):
        return False
    
    return True

def _score(pos, radii):
    return np.sum(radii)

def _local_search(pos, radii, deadline, rng, max_iters=20000, initial_temp=0.005, cooling=0.9995):
    """
    Feasibility-preserving local search.
    Only accepts moves that result in a valid configuration.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    current_pos = pos.copy()
    current_radii = radii.copy()
    current_score = best_score
    
    temp = initial_temp
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Select a random subset of circles to move
        # Moving 3-5 circles allows for more complex rearrangements
        n_move = rng.randint(2, 5)
        indices = rng.choice(N, n_move, replace=False)
        
        new_pos = current_pos.copy()
        new_radii = current_radii.copy()
        
        # Perturb positions and radii
        # Step size decays with temperature
        step_pos = 0.02 * temp + 0.001
        step_rad = 0.01 * temp + 0.001
        
        for idx in indices:
            # Random walk in position
            new_pos[idx, 0] += rng.uniform(-step_pos, step_pos)
            new_pos[idx, 1] += rng.uniform(-step_pos, step_pos)
            
            # Random walk in radius (multiplicative to keep scale)
            # Bias slightly towards increasing radius to maximize sum
            factor = 1.0 + rng.uniform(-step_rad, step_rad * 1.2)
            new_radii[idx] *= factor
            
            # Clamp radii to reasonable bounds
            new_radii[idx] = np.clip(new_radii[idx], 1e-5, 0.4)
            
        # Check validity
        if _is_valid(new_pos, new_radii, tol=1e-8):
            new_score = _score(new_pos, new_radii)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                # Always accept improvement
                current_pos = new_pos
                current_radii = new_radii
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
                    best_radii = new_radii.copy()
            else:
                # Accept worse move with probability exp(delta/temp)
                # This allows escaping local minima while staying valid
                if temp > 1e-6:
                    if rng.uniform() < np.exp(delta / temp):
                        current_pos = new_pos
                        current_radii = new_radii
                        current_score = new_score
        
        temp *= cooling
        
    return best_pos, best_radii, best_score

def _polish(pos, radii, deadline):
    """
    High-precision polish using SLSQP with hard constraints.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.4))
        
    def constraint_boundary(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        # p_i - r_i >= 0
        # 1 - p_i - r_i >= 0
        c1 = p[:, 0] - r
        c2 = 1.0 - p[:, 0] - r
        c3 = p[:, 1] - r
        c4 = 1.0 - p[:, 1] - r
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        dx = p[_II, 0] - p[_JJ, 0]
        dy = p[_II, 1] - p[_JJ, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[_II] + r[_JJ]
        # dist_sq - min_dist^2 >= 0
        return dist_sq - min_dist * min_dist

    def objective(x):
        return -np.sum(x[2 * N:])

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    try:
        # Use SLSQP for final precision
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'disp': False})
        
        if res.success or (res.fun < objective(x0)):
            pos_curr = res.x[:2 * N].reshape(N, 2)
            rad_curr = np.maximum(res.x[2 * N:], 1e-5)
            
            # Verify validity
            if _is_valid(pos_curr, rad_curr, tol=1e-8):
                return pos_curr, rad_curr
    except Exception:
        pass
        
    return pos, radii

def _init_hexagonal():
    """Generate a hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.1
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(8):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        if y + r > 1.0:
            continue
        for col in range(8):
            x = col * dx + offset + r
            if x + r > 1.0:
                continue
            if k >= N:
                break
            pos[k] = [x, y]
            radii[k] = r
            k += 1
        if k >= N:
            break
            
    # Fill remaining with small circles if any
    rng = np.random.RandomState(42)
    while k < N:
        # Place in random gaps
        for _ in range(1000):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            max_r = min(x, 1-x, y, 1-y)
            for j in range(k):
                dist = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                max_r = min(max_r, dist - radii[j])
            if max_r > 0.02:
                pos[k] = [x, y]
                radii[k] = max_r * 0.95
                k += 1
                break
        if k < N:
             pos[k] = [0.5, 0.5]
             radii[k] = 0.01
             k += 1
             
    return pos, radii

def _init_random_greedy(seed=0):
    """Generate a random greedy initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large circle in center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.2
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r = min(x, 1-x, y, 1-y)
            for j in range(i):
                dist = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                max_r = min(max_r, dist - radii[j])
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Hexagonal
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    # 2. Random Greedy 1
    pos_rg1, rad_rg1 = _init_random_greedy(seed=1)
    inits.append((pos_rg1, rad_rg1))
    
    # 3. Random Greedy 2
    pos_rg2, rad_rg2 = _init_random_greedy(seed=2)
    inits.append((pos_rg2, rad_rg2))
    
    # 4. Grid
    pos_grid = np.zeros((N, 2))
    rad_grid = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos_grid[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                rad_grid[k] = 0.09
                k += 1
    inits.append((pos_grid, rad_grid))
    
    rng = np.random.RandomState(123)
    
    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Ensure initial validity
        if not _is_valid(pos_init, rad_init, tol=1e-8):
            # Fallback: shrink all radii
            scale = 0.8
            while not _is_valid(pos_init, rad_init * scale, tol=1e-8) and scale > 0.1:
                scale *= 0.9
            rad_init = rad_init * scale
            
        # Local Search
        pos_ls, rad_ls, score_ls = _local_search(pos_init, rad_init, deadline, rng, max_iters=10000)
        
        if score_ls > best_score:
            best_score = score_ls
            best_pos = pos_ls
            best_radii = rad_ls
            
    # Final Polish on the best result
    if best_pos is not None and time.time() < deadline - 5:
        pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_pol, rad_pol, tol=1e-8):
            score_pol = _score(pos_pol, rad_pol)
            if score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol
                best_radii = rad_pol
                
    # Fallback if no valid solution found
    if best_pos is None:
        best_pos = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            best_pos[i] = [0.5, 0.5]
            best_radii[i] = 0.01
        # Simple grid fallback
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    best_pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    best_radii[k] = 0.09
                    k += 1
        best_score = _score(best_pos, best_radii)
        
    # Final safety check and correction
    if not _is_valid(best_pos, best_radii, tol=1e-9):
        # Aggressive shrink
        scale = 0.9
        while not _is_valid(best_pos, best_radii * scale, tol=1e-9) and scale > 0.1:
            scale *= 0.9
        best_radii = best_radii * scale
        best_score = _score(best_pos, best_radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
