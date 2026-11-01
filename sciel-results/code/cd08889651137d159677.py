import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START

def _compute_energy(vars, penalty_weight=1000.0, boundary_weight=1000.0):
    """
    Compute the energy of the configuration.
    vars: array of shape (3N,) containing [x0, y0, r0, x1, y1, r1, ...]
    Returns: scalar energy value.
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    energy = -np.sum(r)
    
    # Pairwise overlaps
    # Compute all pairwise distances
    # diff[i, j] = xi - xj
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    # r_sum[i, j] = ri + rj
    r_sum = r[:, None] + r[None, :]
    
    # Violations: where dist < r_sum
    # We only care about upper triangle to avoid double counting, 
    # but for vectorized penalty, we can sum all and divide by 2, 
    # or just use a mask.
    # Let's use a mask for i < j
    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    rs_pairs = r_sum[iu]
    
    violations = np.maximum(0, rs_pairs - d_pairs)
    energy += penalty_weight * np.sum(violations**2)
    
    # Boundary violations
    # x < r, x + r > 1, y < r, y + r > 1
    v_left = np.maximum(0, r - x)
    v_right = np.maximum(0, x + r - 1.0)
    v_bottom = np.maximum(0, r - y)
    v_top = np.maximum(0, y + r - 1.0)
    
    energy += boundary_weight * (np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2))
    
    return energy

def _sa_step(vars, T, rng, step_size=0.01):
    """
    Perform one step of Simulated Annealing.
    Returns new vars.
    """
    new_vars = vars.copy()
    # Pick a random circle
    i = rng.randint(N)
    
    # Perturb x, y, r
    # Adaptive step size could be used, but fixed is simpler
    new_vars[3*i] += rng.uniform(-step_size, step_size)
    new_vars[3*i+1] += rng.uniform(-step_size, step_size)
    # Perturb radius slightly, but keep it positive
    dr = rng.uniform(-step_size*0.5, step_size*0.5)
    new_vars[3*i+2] = max(0.01, new_vars[3*i+2] + dr)
    
    # Clip positions to [0, 1] to keep them in bounds roughly
    new_vars[3*i] = np.clip(new_vars[3*i], 0.0, 1.0)
    new_vars[3*i+1] = np.clip(new_vars[3*i+1], 0.0, 1.0)
    
    return new_vars

def _simulated_annealing(init_vars, deadline, rng_seed=42):
    """
    Run Simulated Annealing.
    """
    rng = np.random.RandomState(rng_seed)
    vars = init_vars.copy()
    
    # Initial energy
    current_energy = _compute_energy(vars)
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-6
    num_iterations = 5000  # Adjust based on time
    
    # We need to estimate how many iterations we can do.
    # _compute_energy is vectorized, so it's fast.
    # Let's measure time for 100 iterations.
    start_t = time.time()
    for _ in range(100):
        _ = _compute_energy(vars)
    elapsed = time.time() - start_t
    avg_time_per_iter = elapsed / 100.0
    
    # Reserve time for final polish
    time_budget = deadline - time.time() - 5.0
    if time_budget <= 0:
        return vars
        
    max_iters = int(time_budget / avg_time_per_iter)
    # Cap iterations to avoid infinite loops if time estimate is off
    max_iters = min(max_iters, 20000)
    
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    best_vars = vars.copy()
    best_energy = current_energy
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        T = T_start * (cooling_rate ** i)
        
        new_vars = _sa_step(vars, T, rng, step_size=0.05)
        new_energy = _compute_energy(new_vars)
        
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0 or (T > 1e-9 and rng.rand() < np.exp(-delta_e / T)):
            vars = new_vars
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_vars = vars.copy()
    
    return best_vars

def _local_polish(vars, deadline):
    """
    Use L-BFGS-B to polish the result from SA.
    """
    from scipy.optimize import minimize
    
    def objective(v):
        return _compute_energy(v, penalty_weight=5000.0, boundary_weight=5000.0)
    
    # Gradient is hard to compute analytically for the max(0, ...) terms easily in vectorized form for scipy,
    # so we use numerical gradient or a derivative-free method.
    # Nelder-Mead is slow. Powell is better.
    # Let's try Powell.
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.01, 0.3)) # r
        
    try:
        result = minimize(
            objective,
            vars,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8}
        )
        if time.time() > deadline:
            return vars
        return result.x
    except Exception:
        return vars

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _grid_init(n):
    side = int(np.ceil(np.sqrt(n)))
    points = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(points) < n:
                px = (i + 0.5) / side
                py = (j + 0.5) / side
                r = 0.5 / side * 0.95
                points.append([px, py])
                radii.append(r)
    return np.array(points), np.array(radii)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_vars = None
    best_energy = float('inf')
    
    # Multiple restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Different initializations
        if restart == 0:
            # Grid
            points, radii = _grid_init(N)
        else:
            # Random
            rng = np.random.RandomState(restart * 100)
            points = np.column_stack((rng.uniform(0.1, 0.9, N), rng.uniform(0.1, 0.9, N)))
            radii = np.full(N, 0.08)
            
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Run SA
        vars_sa = _simulated_annealing(x0, deadline, rng_seed=restart*123)
        
        # Polish
        vars_polished = _local_polish(vars_sa, deadline)
        
        # Evaluate
        energy = _compute_energy(vars_polished, penalty_weight=10000.0, boundary_weight=10000.0)
        
        if energy < best_energy:
            best_energy = energy
            best_vars = vars_polished.copy()
            
    if best_vars is None:
        # Fallback
        points, radii = _grid_init(N)
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        best_vars = x0
        
    # Final Validation and Repair
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    if not _is_valid(x_best, y_best, r_best, tol=1e-5):
        # Try to shrink radii slightly to ensure validity
        # This might lower the score, but ensures validity
        scale = 1.0
        while not _is_valid(x_best, y_best, r_best * scale, tol=1e-5) and scale > 0.5:
            scale -= 0.01
        r_best = r_best * scale
        
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
