"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N_CIRCLES = 200
HALF_SIDE = 0.5
KNOWN_BEST_R = 0.036612798904


def _compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    n = len(centers)
    if n != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers))
    if min_dist_to_wall <= 0:
        return 0.0
    
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=2 * 0.04)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    return min(min_dist_to_wall, min_pair_dist / 2)


def _energy(centers, beta=500.0):
    """Soft-min energy: negative of soft-min radius. Lower is better."""
    n = len(centers)
    # Wall distances
    wall_dists = HALF_SIDE - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances (using KD-tree for efficiency)
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=2 * 0.04)
    if len(pairs) == 0:
        min_pair = 1e9
    else:
        dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
        min_pair = np.min(dists)
    
    # Soft-min of wall distance and half pair distance
    vals = np.array([min_wall, min_pair / 2.0])
    # log-sum-exp trick for soft-min
    m = np.min(vals)
    soft_min = m + np.log(np.sum(np.exp(-(vals - m) * beta)) / len(vals)) / beta
    
    return -soft_min  # negative so minimization maximizes radius


def _energy_grad(centers, beta=500.0):
    """Gradient of the soft-min energy."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Wall distances
    wall_dists = HALF_SIDE - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=2 * 0.04)
    if len(pairs) == 0:
        min_pair = 1e9
        pair_grads = {}
    else:
        dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
        min_pair = np.min(dists)
        pair_grads = {}
        for idx, (i, j) in enumerate(pairs):
            diff = centers[i] - centers[j]
            d = dists[idx]
            if d < 1e-10:
                continue
            unit = diff / d
            pair_grads[(i, j)] = (unit, d)
    
    # Soft-min weights
    vals = np.array([min_wall, min_pair / 2.0])
    m = np.min(vals)
    exps = np.exp(-(vals - m) * beta)
    weights = exps / np.sum(exps)
    
    # Gradient of wall soft-min part
    # d(min_wall)/d(centers[k]) = -sign(centers[k]) for the argmin component
    # For soft-min, weight by exp
    for k in range(n):
        for dim in range(2):
            # Wall contribution
            w_k = HALF_SIDE - np.abs(centers[k, dim])
            exp_w = np.exp(-(w_k - m) * beta)
            weight_w = exp_w / np.sum(np.exp(-(vals - m) * beta))
            
            # d(wall_dist)/d(centers[k,dim]) = -sign(centers[k,dim])
            if abs(centers[k, dim]) > 1e-10:
                sign = np.sign(centers[k, dim])
            else:
                sign = 0.0
            grad[k, dim] += -weight_w * (-sign)  # negative of derivative of -soft_min
    
    # Pair contribution
    if len(pairs) > 0:
        for idx, (i, j) in enumerate(pairs):
            d = dists[idx]
            exp_p = np.exp(-(d / 2.0 - m) * beta)
            weight_p = exp_p / np.sum(np.exp(-(vals - m) * beta))
            
            diff = centers[i] - centers[j]
            if d < 1e-10:
                continue
            unit = diff / d
            
            # d(d/2)/d(centers[i]) = unit/2, d(d/2)/d(centers[j]) = -unit/2
            grad[i] += -weight_p * (unit / 2.0)
            grad[j] += -weight_p * (-unit / 2.0)
    
    return grad


def _initialize_hexagonal():
    """Generate hexagonal close-packed initialization."""
    # Try different hexagonal arrangements
    best_config = None
    best_r = 0.0
    
    for m in range(13, 18):
        for n in range(13, 18):
            if m * n < N_CIRCLES:
                continue
            
            # Hexagonal grid
            dx = 1.0 / m
            dy = dx * np.sqrt(3) / 2
            points = []
            for j in range(n):
                for i in range(m):
                    x = -0.5 + dx/2 + i * dx + (j % 2) * dx/2
                    y = -0.5 + dy/2 + j * dy
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        points.append([x, y])
            
            if len(points) < N_CIRCLES:
                continue
            
            # Shuffle and take first N
            np.random.shuffle(points)
            centers = np.array(points[:N_CIRCLES])
            
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_config = centers.copy()
    
    if best_config is not None:
        return best_config
    
    # Fallback: simple grid
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]


def _local_optimize(centers, max_iters=200, step_size=0.005):
    """Gradient-based local optimization using soft-min energy."""
    current = centers.copy()
    current_energy = _energy(current)
    best = current.copy()
    best_energy = current_energy
    
    lr = step_size
    for it in range(max_iters):
        grad = _energy_grad(current)
        
        # Gradient descent step
        new_centers = current - lr * grad
        new_centers = np.clip(new_centers, -HALF_SIDE + 1e-10, HALF_SIDE - 1e-10)
        
        new_energy = _energy(new_centers)
        
        if new_energy < current_energy:
            current = new_centers
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
            # Reduce learning rate slowly
            lr *= 0.995
        else:
            # Reduce learning rate and retry
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return best, _compute_radius(best)


def _slsqp_polish(centers, max_iter=100):
    """SLSQP polish: maximize radius subject to constraints."""
    # Use SLSQP to maximize t subject to:
    # |c_i - c_j| >= 2t for all i<j
    # 0.5 - |c_ik| >= t for all i,k
    
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Minimize negative radius (i.e., maximize radius)
        r = _compute_radius(c)
        return -r
    
    def objective_grad(x):
        c = x.reshape(-1, 2)
        g = _energy_grad(c, beta=200.0)
        return g.flatten()
    
    # Constraints: pairwise distances >= 2t, wall distances >= t
    # We'll use inequality constraints: g(x) >= 0
    
    constraints = []
    
    # Wall constraints: 0.5 - |c_ik| - t >= 0
    # We'll handle these via bounds and a custom approach
    
    # Use a simpler approach: just do gradient-based with projection
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for it in range(max_iter):
        # Find the most violated constraint
        tree = cKDTree(current)
        pairs = tree.query_pairs(r=2 * (current_r + 0.002))
        
        min_pair_dist = 1e9
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(current[i] - current[j])
            if d < min_pair_dist:
                min_pair_dist = d
                min_pair = (i, j)
        
        min_wall = np.min(HALF_SIDE - np.abs(current))
        min_wall_idx = np.argmin(HALF_SIDE - np.abs(current))
        
        # Determine which constraint is most binding
        if min_pair_dist / 2 < min_wall:
            # Pair constraint is binding
            i, j = min_pair
            diff = current[i] - current[j]
            d = min_pair_dist
            if d < 1e-10:
                continue
            unit = diff / d
            push = (2 * current_r - d) / 2 + 0.0005
            if push > 0:
                current[i] += push * unit
                current[j] -= push * unit
        else:
            # Wall constraint is binding
            idx = min_wall_idx // 2
            dim = min_wall_idx % 2
            # Push away from wall
            if current[idx, dim] > 0:
                current[idx, dim] -= 0.001
            else:
                current[idx, dim] += 0.001
        
        current = np.clip(current, -HALF_SIDE, HALF_SIDE)
        
        new_r = _compute_radius(current)
        if new_r > best_r:
            best_r = new_r
            best = current.copy()
        else:
            break
    
    return best, best_r


def _polish(centers, deadline):
    """Multi-stage polish: local optimize then SLSQP."""
    if time.time() > deadline - 10.0:
        return centers, _compute_radius(centers)
    
    # Stage 1: gradient-based local optimization
    result, r = _local_optimize(centers, max_iters=300, step_size=0.008)
    
    if time.time() > deadline - 5.0:
        return result, r
    
    # Stage 2: SLSQP-style constraint projection
    result2, r2 = _slsqp_polish(result, max_iter=200)
    
    if r2 > r:
        return result2, r2
    
    return result, r


def _critical_polish(centers, deadline):
    """Final high-precision polish with tighter tolerances."""
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)
    
    # Very fine local optimization
    result, r = _local_optimize(centers, max_iters=500, step_size=0.002)
    
    # Additional constraint projection rounds
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
        
        tree = cKDTree(result)
        pairs = tree.query_pairs(r=2 * (_compute_radius(result) + 0.001))
        
        improved = False
        for i, j in pairs:
            diff = result[i] - result[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            unit = diff / d
            r_cur = _compute_radius(result)
            if d < 2 * r_cur:
                push = (2 * r_cur - d) / 2 + 1e-5
                result[i] += push * unit
                result[j] -= push * unit
                improved = True
        
        # Wall constraints
        for k in range(N_CIRCLES):
            for dim in range(2):
                w = HALF_SIDE - np.abs(result[k, dim])
                if w < _compute_radius(result):
                    if result[k, dim] > 0:
                        result[k, dim] -= 1e-5
                    else:
                        result[k, dim] += 1e-5
                    improved = True
        
        result = np.clip(result, -HALF_SIDE, HALF_SIDE)
        
        if not improved:
            break
    
    return result, _compute_radius(result)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Initialize with hexagonal pattern
    best_config = _initialize_hexagonal()
    best_r = _compute_radius(best_config)
    
    # Try multiple initializations
    for trial in range(10):
        if time.time() > deadline - 20.0:
            break
        
        # Add noise to hexagonal init
        noise = 0.01 * (1 - trial / 10.0)
        init = best_config + np.random.uniform(-noise, noise, size=(N_CIRCLES, 2))
        init = np.clip(init, -HALF_SIDE, HALF_SIDE)
        
        # Local optimize
        result, r = _local_optimize(init, max_iters=200, step_size=0.006)
        
        if r > best_r:
            best_r = r
            best_config = result.copy()
    
    # Polish the best configuration
    if time.time() < deadline - 15.0:
        polished, pr = _polish(best_config, deadline)
        if pr > best_r:
            best_r = pr
            best_config = polished.copy()
    
    # Critical final polish
    if time.time() < deadline - 5.0:
        final, fr = _critical_polish(best_config, deadline)
        if fr > best_r:
            best_r = fr
            best_config = final.copy()
    
    # Ensure validity
    best_config = np.clip(best_config, -HALF_SIDE, HALF_SIDE)
    
    return best_config
# EVOLVE-BLOCK-END
