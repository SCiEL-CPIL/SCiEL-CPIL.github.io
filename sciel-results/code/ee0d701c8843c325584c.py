"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Multi-start initialization with jittered grids.
    2. Soft-min gradient ascent with adaptive momentum and simulated annealing.
    3. Robust pairwise polish to resolve local conflicts.
    4. Dynamic time allocation between exploration and exploitation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_radius = 0.0
    best_pts = None
    
    # Define a few different initializations to escape local minima
    # 1. Standard 5x5x4 grid with seed 42
    # 2. 5x5x4 grid with seed 123
    # 3. 5x5x4 grid with seed 999
    # 4. Random uniform initialization (coarse)
    
    seeds = [42, 123, 999]
    use_random_start = True
    
    # Time budget allocation:
    # If we have 115s, we can do 3 quick explorations (20s each) and 1 long refinement (35s) + polish (10s)
    # Or just 2 explorations and 1 long refinement.
    # Let's do 3 explorations of ~25s each, keeping the best, then refine the best for ~40s.
    
    num_starts = 3
    exploration_time = 25.0
    refinement_time = 40.0
    polish_time = 5.0
    
    for i in range(num_starts):
        if time.time() > deadline - refinement_time - polish_time - 5.0:
            break
            
        # Initialization
        if i < len(seeds):
            seed = seeds[i]
            nx, ny, nz = 5, 5, 4
            xs = np.linspace(-0.4, 0.4, nx)
            ys = np.linspace(-0.4, 0.4, ny)
            zs = np.linspace(-0.4, 0.4, nz)
            base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
            np.random.seed(seed)
            current_pts = base_points + np.random.uniform(-0.03, 0.03, (N, 3))
        else:
            # Random start
            np.random.seed(0)
            current_pts = np.random.uniform(-0.45, 0.45, (N, 3))
        
        current_pts = np.clip(current_pts, -0.5, 0.5)
        
        # Run optimization
        pts, radius = optimize(current_pts, time_limit=exploration_time, start_time=time.time())
        
        if radius > best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Refine the best found so far
    if best_pts is not None and time.time() < deadline - polish_time - 2.0:
        # Run a longer, finer optimization on the best candidate
        pts, radius = optimize(best_pts, time_limit=refinement_time, start_time=time.time(), fine=True)
        if radius > best_radius:
            best_radius = radius
            best_pts = pts
    
    # Final Polish
    if best_pts is not None and time.time() < deadline - 1.0:
        best_pts, best_radius = polish(best_pts, best_radius, time_limit=polish_time, start_time=time.time())
        
    # Final verification
    if best_pts is None:
        # Fallback: simple grid
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        best_pts = np.array([[x, y, z] for z in zs for y in ys for x in xs])
        best_radius = compute_radius(best_pts)
        
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_pts

def compute_radius(pts):
    """Compute the maximum valid radius for given points (vectorized)."""
    if not np.all(np.abs(pts) <= 0.5):
        return 0.0
    
    n = len(pts)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    boundary_dists = 0.5 - np.abs(pts)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def compute_radius_and_crit(pts):
    """Compute radius and return the critical pair/boundary indices."""
    if not np.all(np.abs(pts) <= 0.5):
        return 0.0, None, None
    
    n = len(pts)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    # Find minimum pair
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_pair_dist = dists[min_pair_idx]
    
    boundary_dists = 0.5 - np.abs(pts)
    min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
    min_boundary = boundary_dists[min_boundary_idx]
    
    if min_pair_dist / 2.0 <= min_boundary:
        r = min_pair_dist / 2.0
        i_idx, j_idx = min_pair_idx
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        crit_pair = (i_idx, j_idx)
        crit_boundary = None
    else:
        r = min_boundary
        crit_pair = None
        crit_boundary = min_boundary_idx
        
    return r, crit_pair, crit_boundary

def soft_min_grad(pts, alpha):
    """
    Compute gradient of soft-min radius w.r.t. point positions.
    Uses log-sum-exp for numerical stability.
    """
    n = len(pts)
    # Pairwise distances
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    pair_diffs = pts[iu[0]] - pts[iu[1]]
    
    # Boundary terms
    boundary_terms = 0.5 - np.abs(pts)
    boundary_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_dists, boundary_flat])
    
    # Log-sum-exp for numerical stability
    neg_alpha_x = -alpha * all_terms
    max_val = np.max(neg_alpha_x)
    exp_vals = np.exp(neg_alpha_x - max_val)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    # Pairwise term gradients
    pair_weights = weights[:len(pair_dists)]
    safe_dists = np.maximum(dists[iu], 1e-10)
    unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    
    # Boundary term gradients
    boundary_weights = weights[len(pair_dists):]
    bw = boundary_weights.reshape(n, 3)
    signs = np.sign(pts)
    grad -= bw * signs
    
    return grad

def optimize(pts, time_limit, start_time, fine=False):
    """
    Run the soft-min gradient ascent optimization.
    """
    current_pts = pts.copy()
    current_radius = compute_radius(current_pts)
    best_radius = current_radius
    best_pts = current_pts.copy()
    
    if fine:
        alpha = 80.0
        lr = 0.005
        T = 0.001
    else:
        alpha = 30.0
        lr = 0.03
        T = 0.005
        
    momentum = np.zeros_like(current_pts)
    beta = 0.9
    no_improve = 0
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        grad = soft_min_grad(current_pts, alpha)
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * grad
        new_pts = current_pts + lr * momentum
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            T *= 0.998
            lr *= 0.9995
            alpha *= 1.001
            if lr < (0.001 if fine else 0.005):
                lr = 0.003 if fine else 0.015
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.97
            lr *= 0.93
            if lr < (0.001 if fine else 0.005):
                if no_improve > (50 if fine else 80):
                    # Perturb to escape local minimum
                    if fine:
                        idx = np.random.choice(N, 3, replace=False)
                        current_pts[idx] += np.random.uniform(-0.01, 0.01, (3, 3))
                    else:
                        idx = np.random.choice(N, 8, replace=False)
                        current_pts[idx] = np.random.uniform(-0.4, 0.4, (8, 3))
                    
                    current_pts = np.clip(current_pts, -0.5, 0.5)
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.005 if fine else 0.03
                    T = 0.001 if fine else 0.005
                    alpha = 80.0 if fine else 30.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.005 if fine else 0.03
                    T = 0.001 if fine else 0.005
                    alpha = 80.0 if fine else 30.0
                    momentum = np.zeros_like(current_pts)
        
        # Track best
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
            
    return best_pts, best_radius

def polish(pts, r, time_limit, start_time):
    """
    High-precision polish: iteratively adjust close pairs and boundaries.
    """
    pts = pts.copy()
    r = r
    deadline = start_time + time_limit
    
    # We iterate and fix the most violated constraints
    # This is a simple iterative repair
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Find all pairs closer than 2*r + epsilon
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Find boundary violations
        boundary_dists = 0.5 - np.abs(pts)
        
        # Check if we are already valid for radius r
        min_pair = np.min(dists)
        min_bound = np.min(boundary_dists)
        
        if min_pair >= 2 * r and min_bound >= r:
            # Try to increase r slightly? No, we just want to ensure validity for the current r
            # Actually, we want to maximize r. So we should try to push them apart to allow a larger r.
            # Let's try to find the critical constraint and push against it.
            pass
        
        # Identify critical pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        if i > j: i, j = j, i
        
        d = pts[i] - pts[j]
        dist = np.linalg.norm(d)
        
        # Identify critical boundary
        min_bound_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        bi, bk = min_bound_idx
        
        # Determine which constraint is tighter relative to current r
        # If pair dist < 2*r, fix pair. If bound dist < r, fix boundary.
        # Actually, we want to maximize the minimum.
        # Let's just try to push the closest pair apart and the closest boundary point in.
        
        # Push pair apart
        if dist < 2 * r * 1.01: # Small tolerance
            if dist > 1e-12:
                unit = d / dist
                # Move them apart to satisfy 2*r
                target_dist = 2 * r
                deficit = target_dist - dist
                if deficit > 0:
                    step = 0.5 * deficit * 0.5 # Damped
                    pts[i] += step * unit
                    pts[j] -= step * unit
        
        # Push boundary point in
        bd = boundary_dists[bi, bk]
        if bd < r * 1.01:
            sign = np.sign(pts[bi, bk])
            if sign == 0: sign = 1
            deficit = r - bd
            if deficit > 0:
                pts[bi, bk] += sign * deficit * 0.5 # Damped
        
        # Clip
        pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Check new radius
        new_r = compute_radius(pts)
        if new_r > r:
            r = new_r
        else:
            # If we can't improve, break to avoid oscillation
            # But maybe a different pair is now critical?
            # Simple heuristic: if no improvement in last few steps, break
            # For now, just break if radius didn't increase
            pass
            
    return pts, r

# EVOLVE-BLOCK-END
