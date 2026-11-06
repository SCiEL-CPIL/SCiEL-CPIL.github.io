"""Optimized packing of 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 110.0  # safety margin before 120s limit
START_TIME = time.time()


def time_left():
    return DEADLINE - (time.time() - START_TIME)


def compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall_dist = np.min(np.minimum(0.5 - np.abs(centers), 
                                   np.stack([0.5 - np.abs(centers), 0.5 - np.abs(centers), 0.5 - np.abs(centers)], axis=-1)), axis=-1)
    wall_r = np.min(wall_dist)
    
    # Pair distances
    # Use chunked computation to save memory
    min_pair_dist = np.inf
    for i in range(N):
        diffs = centers[i+1:] - centers[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair_dist = min(min_pair_dist, np.min(dists))
    
    pair_r = min_pair_dist / 2.0 if min_pair_dist < np.inf else 0.5
    return min(wall_r, pair_r)


def smooth_objective(centers, beta=50.0):
    """
    Smooth objective: we want to maximize r = min(wall_r, pair_r).
    We minimize -r using a smooth approximation.
    Actually, let's use: minimize sum of violations + negative of min distance.
    
    Better approach: minimize F = -min_r where min_r is approximated smoothly.
    Or: maximize the minimum of all constraints.
    
    Let's use: minimize -log(sum(exp(-beta * d_i))) / beta where d_i are all distances (pair and wall)
    This approximates the minimum distance.
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall distances: for each point, distance to nearest wall
    wall_dists = []
    for i in range(N):
        c = centers[i]
        # Distance to each of 6 walls
        d = np.min([0.5 - c[0], c[0] + 0.5, 0.5 - c[1], c[1] + 0.5, 0.5 - c[2], c[2] + 0.5])
        wall_dists.append(d)
    wall_dists = np.array(wall_dists)
    
    # Pair distances
    pair_dists = []
    for i in range(N):
        for j in range(i+1, N):
            pair_dists.append(np.linalg.norm(centers[i] - centers[j]))
    pair_dists = np.array(pair_dists)
    
    # All distances
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Smooth minimum approximation
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    if len(all_dists) == 0:
        return 0.0
    
    # Clip to avoid overflow
    all_dists_clipped = np.maximum(all_dists, 1e-10)
    scaled = -beta * all_dists_clipped
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_min = -log_sum / beta
    
    return -smooth_min  # We minimize this, so we maximize smooth_min


def objective_and_grad(centers, beta=50.0):
    """Objective and gradient for smooth optimization."""
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall distances
    wall_dists = np.zeros(N)
    for i in range(N):
        c = centers[i]
        wall_dists[i] = np.min([0.5 - c[0], c[0] + 0.5, 0.5 - c[1], c[1] + 0.5, 0.5 - c[2], c[2] + 0.5])
    
    # Pair distances
    pair_dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            pair_dists[i, j] = d
            pair_dists[j, i] = d
    
    # Build all distances and their gradients
    # Wall gradients
    wall_grads = np.zeros((N, 3))
    for i in range(N):
        c = centers[i]
        # Gradient of min of linear functions
        # d(0.5-c[k])/dc[k] = -1, d(c[k]+0.5)/dc[k] = 1
        dists = np.array([0.5 - c[0], c[0] + 0.5, 0.5 - c[1], c[1] + 0.5, 0.5 - c[2], c[2] + 0.5])
        grads = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=np.float64)
        min_idx = np.argmin(dists)
        wall_grads[i] = grads[min_idx]
    
    # Pair gradients
    pair_grads = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d > 1e-10:
                grad_i = diff / d
                grad_j = -diff / d
            else:
                grad_i = np.array([0.0, 0.0, 0.0])
                grad_j = np.array([0.0, 0.0, 0.0])
            pair_grads[i, j] = grad_i
            pair_grads[j, i] = grad_j
    
    # Smooth min with gradients
    beta_val = beta
    # We use: f = -1/beta * log(sum(exp(-beta * d_k)))
    # df/dx = -1/beta * (sum(-beta * exp(-beta*d_k) * dd_k/dx) / sum(exp(-beta*d_k)))
    #       = sum(exp(-beta*d_k) * dd_k/dx) / sum(exp(-beta*d_k))
    
    # Wall part
    exp_wall = np.exp(-beta_val * np.maximum(wall_dists, 1e-10))
    sum_exp_wall = np.sum(exp_wall)
    
    # Pair part
    exp_pair = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            exp_pair[i, j] = np.exp(-beta_val * np.maximum(pair_dists[i, j], 1e-10))
            exp_pair[j, i] = exp_pair[i, j]
    sum_exp_pair = np.sum(exp_pair)
    
    total_sum_exp = sum_exp_wall + sum_exp_pair
    
    # Objective value
    if total_sum_exp < 1e-300:
        obj = 0.0
    else:
        obj = -np.log(total_sum_exp) / beta_val
    
    # Gradient
    grad = np.zeros((N, 3))
    
    # Wall contribution
    for i in range(N):
        weight = exp_wall[i] / total_sum_exp
        grad[i] += weight * wall_grads[i]
    
    # Pair contribution
    for i in range(N):
        for j in range(i+1, N):
            weight_ij = exp_pair[i, j] / total_sum_exp
            weight_ji = exp_pair[j, i] / total_sum_exp
            # Each pair counted once in exp_pair[i,j] for i<j, but we need to be careful
            # Actually exp_pair has both [i,j] and [j,i], so sum counts each pair twice
            # Let's recompute: total pair sum should be sum over i<j
            pass
    
    # Recompute pair gradient more carefully
    grad_pair = np.zeros((N, 3))
    sum_exp_pair_correct = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = pair_dists[i, j]
            e = np.exp(-beta_val * np.maximum(d, 1e-10))
            sum_exp_pair_correct += e
    
    total_sum_exp_correct = sum_exp_wall + sum_exp_pair_correct
    
    for i in range(N):
        for j in range(i+1, N):
            d = pair_dists[i, j]
            e = np.exp(-beta_val * np.maximum(d, 1e-10))
            weight = e / total_sum_exp_correct
            diff = centers[i] - centers[j]
            if d > 1e-10:
                unit = diff / d
            else:
                unit = np.array([0.0, 0.0, 0.0])
            grad_pair[i] += weight * unit
            grad_pair[j] += weight * (-unit)
    
    # Wall gradient
    grad_wall = np.zeros((N, 3))
    for i in range(N):
        weight = exp_wall[i] / total_sum_exp_correct
        grad_wall[i] = weight * wall_grads[i]
    
    grad = grad_wall + grad_pair
    
    return -obj, -grad  # We minimize -obj, so return -obj and -grad


def bcc_lattice():
    """Generate BCC lattice points."""
    # BCC: simple cubic + body center
    # In a unit cube, BCC has 2 atoms per cell
    # For 100 spheres, we need about 50 cells
    # Let's use a 4x4x4 grid of BCC cells = 128 points, take 100
    points = []
    m = 4
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                points.append([x, y, z])
                # Body center offset
                points.append([x + 0.5/m, y + 0.5/m, z + 0.5/m])
    points = np.array(points[:N])
    return points


def fcc_lattice():
    """Generate FCC-like lattice points."""
    # FCC: corners + face centers
    # For simplicity, use a distorted grid
    points = []
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = ticks[i]
                y = ticks[j]
                z = ticks[k]
                # Add small perturbations based on parity
                if (i + j + k) % 2 == 0:
                    x += 0.1 / m
                    y += 0.1 / m
                else:
                    x -= 0.1 / m
                    z += 0.1 / m
                points.append([x, y, z])
    points = np.array(points[:N])
    # Clip to cube
    points = np.clip(points, -0.49, 0.49)
    return points


def random_start():
    """Random start within the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def optimize_from_start(start, max_time=None):
    """Run optimization from a given start."""
    if max_time is None:
        max_time = time_left() - 5.0
    if max_time <= 0:
        return start
    
    best_centers = start.copy()
    best_radius = compute_radius(start)
    
    # Multi-start with decreasing beta
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time_left() < 10.0:
            break
        
        # Try multiple initializations
        for attempt in range(3):
            if time_left() < 10.0:
                break
            
            # Add small random perturbation
            if attempt == 0:
                current = start.copy()
            else:
                current = start + np.random.normal(0, 0.05, size=(N, 3))
                current = np.clip(current, -0.48, 0.48)
            
            # Run L-BFGS-B
            result = minimize(
                smooth_objective,
                current,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * 3 * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                candidate = result.x
                r = compute_radius(candidate)
                if r > best_radius:
                    best_radius = r
                    best_centers = candidate.copy()
    
    return best_centers


def final_polish(centers, max_time=None):
    """Final penalty-based polish to tighten constraints."""
    if max_time is None:
        max_time = time_left() - 5.0
    if max_time <= 0:
        return centers
    
    best_centers = centers.copy()
    best_radius = compute_radius(centers)
    
    # Use a penalty method: minimize sum of max(0, r_target - d_i)^2 + negative margin
    # Or simply do a few more L-BFGS steps with high beta
    
    for beta in [200.0, 500.0]:
        if time_left() < 5.0:
            break
        
        result = minimize(
            smooth_objective,
            best_centers,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * 3 * N,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < 0:
            candidate = result.x
            r = compute_radius(candidate)
            if r > best_radius:
                best_radius = r
                best_centers = candidate.copy()
    
    return best_centers


def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Generate multiple diverse starts
    starts = [
        bcc_lattice(),
        fcc_lattice(),
        random_start()
    ]
    
    # Add more random starts
    for _ in range(3):
        if time_left() < 20.0:
            break
        starts.append(random_start())
    
    best_centers = None
    best_radius = -np.inf
    
    for i, start in enumerate(starts):
        if time_left() < 15.0:
            break
        
        centers = optimize_from_start(start)
        r = compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final polish
    if time_left() > 10.0:
        best_centers = final_polish(best_centers)
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
