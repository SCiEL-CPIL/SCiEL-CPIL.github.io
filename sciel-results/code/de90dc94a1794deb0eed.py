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
    1. Multi-start initialization with jittered grids and random starts.
    2. Soft-min gradient ascent with adaptive momentum and simulated annealing.
    3. Robust pairwise polish to resolve local conflicts.
    4. Dynamic time allocation between exploration and exploitation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_radius = 0.0
    best_pts = None
    
    # Define a few different initializations to escape local minima
    seeds = [42, 123, 999, 7, 555]
    
    # Time budget allocation:
    # 3 quick explorations (~20s each) + 1 long refinement (~40s) + polish (~5s)
    num_starts = 3
    exploration_time = 22.0
    refinement_time = 40.0
    polish_time = 8.0
    
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
        alpha = 100.0
        lr = 0.003
        T = 0.0005
    else:
        alpha = 40.0
        lr = 0.02
        T = 0.003
        
    momentum = np.zeros_like(current_pts)
    beta = 0.85
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
            T *= 0.999
            lr *= 0.9998
            alpha *= 1.0005
            if lr < (0.0005 if fine else 0.003):
                lr = 0.002 if fine else 0.01
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.98
            lr *= 0.95
            if lr < (0.0005 if fine else 0.003):
                if no_improve > (80 if fine else 120):
                    # Perturb to escape local minimum
                    if fine:
                        idx = np.random.choice(N, 5, replace=False)
                        current_pts[idx] += np.random.uniform(-0.005, 0.005, (5, 3))
                    else:
                        idx = np.random.choice(N, 10, replace=False)
                        current_pts[idx] = np.random.uniform(-0.4, 0.4, (10, 3))
                    
                    current_pts = np.clip(current_pts, -0.5, 0.5)
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.003 if fine else 0.02
                    T = 0.0005 if fine else 0.003
                    alpha = 100.0 if fine else 40.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.003 if fine else 0.02
                    T = 0.0005 if fine else 0.003
                    alpha = 100.0 if fine else 40.0
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
    no_improve_count = 0
    max_no_improve = 50
    
    for _ in range(2000):
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
        
        # Identify critical pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        if i > j: i, j = j, i
        
        d = pts[i] - pts[j]
        dist = np.linalg.norm(d)
        
        # Identify critical boundary
        min_bound_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        bi, bk = min_bound_idx
        
        # Determine which constraint is tighter
        pair_constraint = dist / 2.0
        bound_constraint = boundary_dists[bi, bk]
        
        improved = False
        
        # Push pair apart if it's the bottleneck
        if pair_constraint < bound_constraint:
            if dist > 1e-12:
                unit = d / dist
                target_dist = 2 * r
                deficit = target_dist - dist
                if deficit > 0:
                    step = 0.3 * deficit
                    pts[i] += step * unit
                    pts[j] -= step * unit
                    improved = True
        
        # Push boundary point in if it's the bottleneck
        if bound_constraint <= pair_constraint:
            bd = boundary_dists[bi, bk]
            if bd < r:
                sign = np.sign(pts[bi, bk])
                if sign == 0: sign = 1
                deficit = r - bd
                if deficit > 0:
                    pts[bi, bk] += sign * deficit * 0.3
                    improved = True
        
        # Clip
        pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Check new radius
        new_r = compute_radius(pts)
        if new_r > r:
            r = new_r
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return pts, r

# EVOLVE-BLOCK-END
