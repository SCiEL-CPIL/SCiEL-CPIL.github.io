"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with high-precision L-BFGS polishing
using a full-pairwise smooth violation objective.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances (Full N x N for accuracy)
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Violation for pair (i,j): softplus(beta * (2*r_target - d_ij))
    Violation for wall i: softplus(beta * (r_target - w_i))
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(dist_sq)
    pair_viol = 2.0 * r_target - dists
    
    # Softplus: log(1 + exp(z)) / beta
    # Clip to avoid overflow in exp
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)
    
    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta
    
    return float(np.sum(soft_wall) + np.sum(soft_pair))

def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))  # sigmoid(beta * viol)
    
    for i in range(N):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        
        # d/dx of wall_dist
        # wall_dist = min(0.5-|x|, 0.5-|y|)
        if wall_dists_x[i] < wall_dists_y[i]:
            # 0.5 - |x| -> derivative is -sign(x)
            grad[2*i] += s * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            # 0.5 - |y| -> derivative is -sign(y)
            grad[2*i+1] += s * np.sign(centers[i, 1])
        else:
            # Tie, split gradient
            grad[2*i] += 0.5 * s * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * s * np.sign(centers[i, 1])
            
    # Pair gradients
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(dist_sq)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    # Only iterate over upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        s = sig_pair[i, j]
        if s < 1e-10:
            continue
        
        d = dists[i, j]
        if d < 1e-12:
            continue
            
        # d/dx_i of (2r - d) is -d/dx_i(d)
        # d/dx_i(d) = (x_i - x_j) / d
        # So d/dx_i (softplus) = sigmoid * beta * (- (x_i - x_j) / d) * (1/beta) = -sigmoid * (x_i - x_j) / d
        # Wait, derivative of log(1+exp(beta*v))/beta is sigmoid(beta*v) * v'
        # v = 2r - d. v' = -d/dx(d) = -(x_i - x_j)/d
        # So grad_i = -sig * (x_i - x_j) / d
        # grad_j = -sig * (x_j - x_i) / d = sig * (x_i - x_j) / d
        
        factor = s / d
        diff_vec = centers[i] - centers[j]
        
        grad[2*i:2*i+2] -= factor * diff_vec
        grad[2*j:2*j+2] += factor * diff_vec
        
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
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
    
    return best_grid if best_grid is not None else _initial_grid()

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.02
    T_end = 0.0001
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick radius check
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # Approximate min pair using KDTree for speed in SA loop
        from scipy.spatial import cKDTree
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        curr_r = min(min_pair, min_wall)
        
        # Move generation
        i = rng.integers(0, N)
        move_size = 0.005 + 0.01 * T / T_start
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        
        trial = current.copy()
        trial[i] += direction * move_size
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize violation for a specific target radius r_target.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Initial bounds
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    
    # We will perform a few iterations of "try to fit r, if success increase lo, else decrease hi"
    # But since L-BFGS is expensive, we do a coarse search first, then fine.
    
    search_steps = 8
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        # Try to find a configuration that fits radius 'mid'
        # Use a moderate beta for this stage
        beta = 200.0
        x0 = best.flatten()
        
        # Time budget per L-BFGS call
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(10.0, remaining * 0.8)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        # Check if we actually achieved the target radius (with tolerance)
        if trial_r >= mid - 1e-7:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using increasing beta.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Progressive sharpening of the violation objective
    betas = [500, 1000, 2000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        # Use current best radius as target (or slightly higher to push)
        # To maximize, we should target the current best radius and see if we can improve it,
        # or target a slightly higher value.
        # Since we are polishing, we target the current best radius to tighten the packing.
        r_target = best_r
        
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(10.0, remaining * 0.8)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    # 1. Initialization
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    
    best_centers = None
    best_radius = 0
    
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # 2. Simulated Annealing (Global Search)
    sa_budget = min(30.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
        
    # 3. Feasibility-Driven Radius Refinement (Coarse to Fine)
    # This step finds the "ceiling" for the current local structure
    if time.time() < deadline - 40:
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=min(40.0, deadline - time.time() - 20)
        )
        
    # 4. High Precision Polish (Final Tightening)
    if time.time() < deadline - 10:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20.0, deadline - time.time() - 5)
        )
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
