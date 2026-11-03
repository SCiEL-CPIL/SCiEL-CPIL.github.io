"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=1000.0):
    """
    Differentiable surrogate for the radius.
    Uses log-sum-exp to approximate the minimum of wall distances and half-pairwise distances.
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise distances (only upper triangle to avoid double counting and self)
    idx = np.triu_indices(N, k=1)
    diff = centers[idx[0]] - centers[idx[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists])
    
    # Soft min: -logsumexp(-beta * dists) / beta
    # To avoid overflow, shift by max
    max_d = np.max(all_dists)
    shifted = -beta * (all_dists - max_d)
    # Clip to prevent exp overflow (though shifted <= 0, so exp <= 1)
    # Actually, if all_dists are small, -beta*(d - max) can be large positive if d < max?
    # No, d <= max, so d - max <= 0, so -beta*(d-max) >= 0.
    # Wait, soft min is approx min(x) ≈ -1/beta * log(sum(exp(-beta * x))).
    # Let's use the stable formulation:
    # min(x) ≈ (1/beta) * logsumexp(-beta * x) ... wait, logsumexp is log(sum(exp(x))).
    # So -1/beta * log(sum(exp(-beta * x))) = -1/beta * logsumexp(-beta * x).
    
    # Let y = -beta * all_dists.
    # We want -1/beta * log(sum(exp(y))) = -1/beta * (max_y + log(sum(exp(y - max_y))))
    y = -beta * all_dists
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return 0.0
    log_sum_exp = max_y + np.log(np.sum(np.exp(y - max_y)))
    soft_min_val = -log_sum_exp / beta
    
    return soft_min_val

def _objective_smooth(x, beta=1000.0):
    """
    Objective function for minimization.
    Minimize -soft_min_radius.
    """
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _local_refine_smooth(centers, time_limit=10.0, beta=1000.0):
    """
    Use L-BFGS-B to refine the configuration using the smooth surrogate.
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    # Box constraints: centers must be in [-0.5, 0.5]
    # But we want them strictly inside to avoid boundary issues in gradient?
    # L-BFGS-B handles bounds well.
    # However, the objective is defined for any x, but physically we want them inside.
    # The soft min will penalize going outside (wall dist becomes negative).
    # So we can set bounds to [-0.5, 0.5] or slightly tighter to be safe.
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 5000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'disp': False
            },
            callback=callback,
            args=(beta,)
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8 (56 pts -> 50)
    c1 = _make_grid_rect(7, 8)
    inits.append(c1)
    
    # 2. Rectangular Grid 8x7
    c2 = _make_grid_rect(8, 7)
    inits.append(c2)
    
    # 3. Hexagonal Grid
    # Estimate spacing for 50 points in unit square.
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # 5. Jittered Grid
    c_jit = _make_grid_rect(7, 8) + np.random.uniform(-0.02, 0.02, size=(N, 2))
    inits.append(c_jit)
    
    # Refine each initial configuration
    # L-BFGS-B is much faster per iteration than Nelder-Mead for smooth functions.
    # We can afford more restarts or tighter tolerances.
    
    # Phase 1: Broad search with moderate beta
    beta_start = 500.0
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        time_left = DEADLINE - 20 - (time.time() - start_time)
        alloc_time = min(15.0, time_left / max(1, len(inits) - i))
        
        refined = _local_refine_smooth(init, time_limit=alloc_time, beta=beta_start)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # Phase 2: Polish the best candidate with higher beta (sharper min approximation)
    if best_centers is not None and time.time() - start_time < DEADLINE - 10:
        time_left = DEADLINE - 10 - (time.time() - start_time)
        # Use a higher beta to make the surrogate closer to the true min
        # This helps fine-tune the positions of the critical pairs
        refined_polish = _local_refine_smooth(best_centers, time_limit=time_left * 0.8, beta=2000.0)
        r_polish = _radius(refined_polish)
        
        if r_polish > best_r:
            best_r = r_polish
            best_centers = refined_polish.copy()
            
    # Phase 3: If time remains, try a few more random restarts with L-BFGS-B
    # L-BFGS-B is fast, so we can try a few quick ones
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 3: break
        
        refined = _local_refine_smooth(c_rand, time_limit=time_left * 0.9, beta=1000.0)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
            # Quick polish on new best
            if time.time() - start_time < DEADLINE - 3:
                time_left = DEADLINE - 3 - (time.time() - start_time)
                refined_polish = _local_refine_smooth(best_centers, time_limit=time_left * 0.7, beta=3000.0)
                r_polish = _radius(refined_polish)
                if r_polish > best_r:
                    best_r = r_polish
                    best_centers = refined_polish.copy()

    return best_centers
