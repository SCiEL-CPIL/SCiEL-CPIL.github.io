"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 115.0  # Safety margin before 120s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, tau=0.01):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree for speed in objective? 
    # No, L-BFGS needs smoothness. cKDTree is not differentiable.
    # We must use the full pairwise distance matrix for the gradient.
    # For N=50, 50*50 = 2500 distances. This is very fast.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only need the upper triangle for unique pairs
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dists[iu, ju] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.01):
    """
    Compute gradient of -soft_min objective analytically (vectorized).
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dists[iu, ju] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min weights
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    wall_weights = weights[:N]
    sx = np.sign(centers[:, 0])
    sy = np.sign(centers[:, 1])
    grad[0::2] += wall_weights * sx
    grad[1::2] += wall_weights * sy
    
    # Pairwise gradient contributions
    pair_weights = weights[N:]
    d_vals = dists[iu, ju]
    valid = d_vals > 1e-12
    factors = np.zeros(len(iu))
    factors[valid] = 0.5 * pair_weights[valid] / d_vals[valid]
    
    diff_ij = centers[iu] - centers[ju]
    weighted_diff = factors[:, np.newaxis] * diff_ij
    
    np.add.at(grad, (iu * 2), -weighted_diff[:, 0])
    np.add.at(grad, (iu * 2 + 1), -weighted_diff[:, 1])
    np.add.at(grad, (ju * 2), weighted_diff[:, 0])
    np.add.at(grad, (ju * 2 + 1), weighted_diff[:, 1])
    
    return grad

def _lbfgs_optimize(x0, tau, time_budget):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 5000, 'disp': False, 'maxls': 50}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine(centers, time_budget):
    """
    Multi-phase refinement with adaptive tau and jittering.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Phase 1: Coarse optimization
    tau = 0.05
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        trial = _lbfgs_optimize(current.flatten(), tau, min(10, remaining * 0.5))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            current = trial
    
    # Phase 2: Medium optimization
    tau = 0.01
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        trial = _lbfgs_optimize(current.flatten(), tau, min(15, remaining * 0.5))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            current = trial
    
    # Phase 3: Fine optimization
    tau = 0.002
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        trial = _lbfgs_optimize(current.flatten(), tau, min(15, remaining * 0.5))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            current = trial
    
    # Phase 4: Jitter and re-optimize (escape local minima)
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    for _ in range(3):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Add small noise
        noise = rng.normal(0, 0.005, size=current.shape)
        jittered = current + noise
        jittered = np.clip(jittered, -0.49, 0.49)
        
        # Refine jittered solution
        trial = _lbfgs_optimize(jittered.flatten(), 0.005, min(10, remaining * 0.5))
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            current = trial
        else:
            # Keep current if jitter didn't help, but maybe try smaller jitter next time
            pass
            
    # Phase 5: Final high-precision polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        tau = 0.0005
        trial = _lbfgs_optimize(best.flatten(), tau, min(10, remaining))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    return best, best_r

def _hex_init(seed=42):
    """Hexagonal packing with different row configurations."""
    rng = np.random.default_rng(seed)
    
    best = None
    best_r = 0
    
    for attempt in range(10):
        n_rows = rng.integers(6, 10)
        base = N // n_rows
        extra = N % n_rows
        rows = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_cols = max(rows)
        spacing_x = 1.0 / (max_cols + 0.5)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        total_height = (n_rows - 1) * spacing_y
        y_start = -0.5 + 0.5 * spacing_y
        if total_height > 1.0 - spacing_y:
            continue
        
        centers = []
        for r_idx, n_cols in enumerate(rows):
            offset = 0.5 * spacing_x if r_idx % 2 == 1 else 0.0
            x_start = -0.5 + spacing_x * 0.5 + offset
            for c in range(n_cols):
                x = x_start + c * spacing_x
                y = y_start + r_idx * spacing_y
                centers.append([x, y])
        
        if len(centers) != N:
            continue
        
        centers = np.array(centers)
        centers = centers + rng.normal(0, 0.003, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best = centers
    
    return best if best is not None else _random_init(99)

def _random_init(seed=99):
    """Random initial positions with repulsion to avoid overlaps."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(N, 2))
    
    # Simple repulsion to spread out
    for _ in range(50):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(centers)
        iu, ju = np.triu_indices(N, k=1)
        for i, j in zip(iu, ju):
            d = dists[i, j]
            if d < 0.1:
                overlap = 0.1 - d
                strength = overlap / d
                diff_ij = centers[i] - centers[j]
                forces[i] += strength * diff_ij
                forces[j] -= strength * diff_ij
        
        centers += 0.1 * forces
        centers = np.clip(centers, -0.45, 0.45)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate initializations
    inits = [
        _hex_init(42),
        _hex_init(123),
        _random_init(99),
        _random_init(100),
        _random_init(101)
    ]
    
    # Select best few initializations
    scored_inits = [(init, _compute_radius(init)) for init in inits]
    scored_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_centers = scored_inits[0][0]
    best_radius = scored_inits[0][1]
    
    # Allocate time to top 3 initializations
    # 1st: 50%, 2nd: 30%, 3rd: 20%
    time_allocs = [0.5, 0.3, 0.2]
    
    for i in range(3):
        if i >= len(scored_inits):
            break
            
        init, init_r = scored_inits[i]
        if init_r <= best_radius and i > 0:
            continue # Skip if not competitive
            
        remaining = deadline - time.time()
        budget = remaining * time_allocs[i]
        
        if budget < 5:
            break
            
        refined, refined_r = _refine(init, budget)
        
        if refined_r > best_radius:
            best_radius = refined_r
            best_centers = refined
    
    # Final global polish on the best result
    remaining = deadline - time.time()
    if remaining > 5:
        final, final_r = _refine(best_centers, remaining)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final
            
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _hex_init(42)
        
    return best_centers
