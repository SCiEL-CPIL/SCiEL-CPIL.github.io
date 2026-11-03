"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
import time
import random

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum packing radius for centers in [-0.5, 0.5]^3."""
    # Distance to walls: min over all coords of (0.5 - |coord|)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = float(np.min(wall_dists))
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(0.5)
    
    min_pair_dist = float('inf')
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if d < min_pair_dist:
            min_pair_dist = d
    
    return min(min_pair_dist, 2.0 * min_wall) / 2.0

def _soft_min_objective_and_grad(centers, beta=200.0):
    """
    Compute a smooth surrogate for the min-distance objective and its gradient.
    Uses log-sum-exp to approximate min of {pair distances, 2*wall distances}.
    Returns (objective_value, gradient).
    """
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    wall_values = 2.0 * wall_dists      # shape (N, 3)
    
    # Pair distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(0.5)
    
    pair_dists = []
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        pair_dists.append(d)
    
    all_dists = np.concatenate([pair_dists, wall_values.ravel()])
    
    # Log-sum-exp for soft min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Wall gradient contribution
    exp_weights_wall = np.exp(-beta * (wall_values - min_val))
    total_weight = np.sum(exp_weights_wall) + np.sum(np.exp(-beta * (np.array(pair_dists) - min_val)))
    
    # Wall part
    for i in range(N):
        for k in range(3):
            w = exp_weights_wall[i, k] / total_weight
            if centers[i, k] > 0:
                grad[i, k] -= 2.0 * w
            else:
                grad[i, k] += 2.0 * w
    
    # Pair part
    exp_weights_pair = np.exp(-beta * (np.array(pair_dists) - min_val))
    for idx, (i, j) in enumerate(pairs):
        d = pair_dists[idx]
        if d < 1e-10:
            continue
        w = exp_weights_pair[idx] / total_weight
        diff = centers[i] - centers[j]
        grad[i] += w * diff / d
        grad[j] -= w * diff / d
    
    return soft_min_val, grad

def _generate_bcc_lattice():
    """Generate a BCC lattice arrangement for N spheres."""
    # BCC: simple cubic + center of each cube
    # For 50 spheres, use a 4x4x4 grid = 64 positions, take first 50
    m = 4
    # Simple cubic points
    sc_points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                sc_points.append([(x + 0.5) / m - 0.5,
                                  (y + 0.5) / m - 0.5,
                                  (z + 0.5) / m - 0.5])
    
    # BCC center points
    bcc_points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                bcc_points.append([(x + 0.5) / m - 0.5 + 0.5/m,
                                   (y + 0.5) / m - 0.5 + 0.5/m,
                                   (z + 0.5) / m - 0.5 + 0.5/m])
    
    # Combine and sort by distance from center to pick best N
    all_points = sc_points + bcc_points
    all_points = np.array(all_points)
    
    # Scale to fit in cube with margin
    max_coord = np.max(np.abs(all_points))
    if max_coord > 0.45:
        all_points *= 0.45 / max_coord
    
    # Pick N points that are most spread out
    # Use greedy: start with point farthest from center, then add farthest from existing
    center = np.zeros(3)
    remaining = list(range(len(all_points)))
    selected = []
    
    # Start with point farthest from center
    dists_to_center = np.linalg.norm(all_points - center, axis=1)
    first = np.argmax(dists_to_center)
    selected.append(first)
    remaining.remove(first)
    
    while len(selected) < N and remaining:
        # Find point farthest from all selected
        min_dists = np.full(len(remaining), np.inf)
        for sel in selected:
            d = np.linalg.norm(all_points[remaining] - all_points[sel], axis=1)
            min_dists = np.minimum(min_dists, d)
        
        next_idx = remaining[np.argmax(min_dists)]
        selected.append(next_idx)
        remaining.remove(next_idx)
    
    return all_points[selected]

def _generate_fcc_lattice():
    """Generate an FCC lattice arrangement for N spheres."""
    # FCC: face-centered cubic
    m = 4  # 4x4x4 unit cells gives many points
    points = []
    
    # Corners of each cell
    for z in range(m):
        for y in range(m):
            for x in range(m):
                # Corner
                points.append([(x + 0.5) / m - 0.5,
                               (y + 0.5) / m - 0.5,
                               (z + 0.5) / m - 0.5])
                # Face centers
                points.append([(x + 0.5) / m - 0.5,
                               (y + 0.5) / m - 0.5 + 0.5/m,
                               (z + 0.5) / m - 0.5])
                points.append([(x + 0.5) / m - 0.5 + 0.5/m,
                               (y + 0.5) / m - 0.5,
                               (z + 0.5) / m - 0.5])
                points.append([(x + 0.5) / m - 0.5,
                               (y + 0.5) / m - 0.5,
                               (z + 0.5) / m - 0.5 + 0.5/m])
    
    points = np.array(points)
    
    # Remove duplicates (corners shared between cells)
    # Scale to fit
    max_coord = np.max(np.abs(points))
    if max_coord > 0.45:
        points *= 0.45 / max_coord
    
    # Deduplicate
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in unique_points):
            unique_points.append(p)
    
    unique_points = np.array(unique_points)
    
    # Pick N most spread out points
    center = np.zeros(3)
    remaining = list(range(len(unique_points)))
    selected = []
    
    dists_to_center = np.linalg.norm(unique_points - center, axis=1)
    first = np.argmax(dists_to_center)
    selected.append(first)
    remaining.remove(first)
    
    while len(selected) < N and remaining:
        min_dists = np.full(len(remaining), np.inf)
        for sel in selected:
            d = np.linalg.norm(unique_points[remaining] - unique_points[sel], axis=1)
            min_dists = np.minimum(min_dists, d)
        
        next_idx = remaining[np.argmax(min_dists)]
        selected.append(next_idx)
        remaining.remove(next_idx)
    
    return unique_points[selected]

def _optimize_from_initial(centers, time_limit, seed=42):
    """Optimize sphere centers using gradient ascent on soft-min objective."""
    rng = np.random.default_rng(seed)
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    # Parameters
    beta = 0.85
    step_size = 0.015
    patience = 40
    no_improve = 0
    beta_lse = 150.0
    
    start_time = time.time()
    prev_centers = centers.copy()
    
    for iteration in range(8000):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        # Compute gradient using soft-min
        _, grad = _soft_min_objective_and_grad(centers, beta_lse)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Nesterov momentum
        v = beta * (prev_centers - centers)
        prev_centers = centers.copy()
        centers_new = centers + step_size * (grad + v)
        
        # Keep within bounds
        centers_new = np.clip(centers_new, -0.495, 0.495)
        
        # Check improvement
        new_score = _compute_radius(centers_new)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers_new.copy()
            no_improve = 0
            step_size = min(step_size * 1.03, 0.04)
        else:
            no_improve += 1
            step_size = max(step_size * 0.96, 0.002)
        
        centers = centers_new
        
        if no_improve > patience:
            break
    
    return best_centers, best_score

def _local_polish(centers, time_limit, seed=99):
    """Fine-tune with very high beta for precise local optimization."""
    best_centers = centers.copy()
    best_score = _compute_radius(centers)
    
    beta_lse = 500.0
    step_size = 0.003
    patience = 30
    no_improve = 0
    
    start_time = time.time()
    
    for iteration in range(3000):
        if time.time() - start_time > time_limit - 1.0:
            break
        
        _, grad = _soft_min_objective_and_grad(centers, beta_lse)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        else:
            break
        
        centers_new = centers + step_size * grad
        centers_new = np.clip(centers_new, -0.499, 0.499)
        
        new_score = _compute_radius(centers_new)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers_new.copy()
            no_improve = 0
        else:
            no_improve += 1
            step_size = max(step_size * 0.9, 0.0005)
        
        centers = centers_new
        
        if no_improve > patience:
            break
    
    return best_centers, best_score

def construct():
    """Construct 50 sphere centers in the unit cube to maximize packing radius."""
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = 0.0
    
    # Generate multiple initializations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(("grid", grid[:N].copy()))
    
    # 2. Jittered grid
    rng_jitter = np.random.default_rng(1)
    inits.append(("jittered_grid", grid[:N].copy() + rng_jitter.normal(0, 0.015, (N, 3))))
    
    # 3. BCC lattice
    try:
        bcc = _generate_bcc_lattice()
        if bcc.shape == (N, 3):
            inits.append(("bcc", bcc))
    except:
        pass
    
    # 4. FCC lattice
    try:
        fcc = _generate_fcc_lattice()
        if fcc.shape == (N, 3):
            inits.append(("fcc", fcc))
    except:
        pass
    
    # 5. Random
    rng_rand = np.random.default_rng(2)
    inits.append(("random", rng_rand.uniform(-0.45, 0.45, (N, 3))))
    
    # Optimize from each initialization
    for name, init_centers in inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 25.0:
            break
        
        time_remaining = time_limit - elapsed - 8.0
        if time_remaining < 5.0:
            break
        
        try:
            opt_centers, opt_score = _optimize_from_initial(init_centers, time_remaining)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers
        except Exception:
            continue
    
    # Local polish on best result
    if best_centers is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 15.0:
            time_remaining = time_limit - elapsed - 3.0
            if time_remaining > 5.0:
                try:
                    polished, polished_score = _local_polish(best_centers, time_remaining)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_centers = polished
                except Exception:
                    pass
    
    # Final safety polish
    if best_centers is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5.0:
            time_remaining = time_limit - elapsed - 1.0
            if time_remaining > 3.0:
                try:
                    polished, polished_score = _local_polish(best_centers, time_remaining, seed=123)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_centers = polished
                except Exception:
                    pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = inits[0][1].copy()
    
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers

# EVOLVE-BLOCK-END
