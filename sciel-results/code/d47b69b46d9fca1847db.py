"""Pack 50 equal circles in a unit square using Graph-Based Violation Minimization + PGD."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n):
    """Create a hexagonal grid of n points centered in the square."""
    # For n=50, a 7x8 or similar grid works.
    # Hexagonal packing:
    # Row 0: x offsets 0, 1, 2...
    # Row 1: x offsets 0.5, 1.5, 2.5...
    
    # Target density: 50 circles in 1x1 square.
    # Area per circle ~ 1/50 = 0.02.
    # Hexagonal area per circle (radius r) = r^2 * 2 * sqrt(3).
    # r^2 * 3.464 = 0.02 => r^2 ~ 0.0057 => r ~ 0.075.
    # Diameter ~ 0.15.
    # Grid spacing ~ 0.15.
    # Number of points across ~ 1/0.15 = 6.6.
    
    # Let's use a denser grid.
    # dx = 1/7. dy = sqrt(3)/14.
    # Rows: 1.0 / (sqrt(3)/14) = 14/1.732 = 8.08 -> 8 rows.
    # Cols: 7.
    # Total ~ 56 points. Good.
    
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter to square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Sort by distance from center, keep closest N?
        # This creates a blob. 
        # Better: Keep a rectangular region in the center.
        # Or just take the first N.
        # Let's try sorting by "potential" - distance to walls.
        # Points closer to walls can potentially expand more?
        # No, let's just take the N points that form the most compact set?
        # For initialization, any valid non-overlapping set is fine.
        # Let's just take the first N.
        points = points[:N]
    elif len(points) < N:
        # Add random points?
        # This shouldn't happen with 8x7 grid.
        pass
        
    return points

def _graph_pgd(centers, r_target, time_limit=5.0, threshold_factor=1.5):
    """
    Minimize sum of squared violations using Projected Gradient Descent.
    Violations:
      1. d_ij < 2r  -> (2r - d_ij)^2
      2. x_i < -0.5 + r or x_i > 0.5 - r -> (boundary violation)^2
      3. y_i < -0.5 + r or y_i > 0.5 - r -> (boundary violation)^2
    
    We use a dynamic graph of pairs where d_ij < threshold * 2r.
    """
    centers = centers.copy()
    r = r_target
    start_time = time.time()
    
    # Learning rate schedule
    lr_init = 0.01
    lr_decay = 0.995
    
    # Threshold for graph edges
    # We care about pairs that are close to violating.
    # If d_ij > 2r + margin, they don't contribute to gradient.
    # Let's use a threshold slightly larger than 2r to be safe.
    edge_threshold = 2.0 * r * threshold_factor
    
    # Precompute indices for potential pairs? 
    # For N=50, 1225 pairs. We can just check all pairs each step if needed,
    # but "graph-based" implies we might skip far ones. 
    # With N=50, full pairwise is fast (1225 ops). 
    # The "graph" aspect here is more about the structure of the objective 
    # being sparse if we were doing something else, but for PGD we compute 
    # the full gradient of the violation function.
    
    # However, to respect the "graph-based" strategy description, 
    # we can maintain a list of active pairs.
    
    active_pairs = []
    
    def get_active_pairs(c, r, thresh):
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < thresh:
                    pairs.append((i, j))
        return pairs
    
    # Initial active pairs
    active_pairs = get_active_pairs(centers, r, edge_threshold)
    
    lr = lr_init
    best_local_r = _radius(centers)
    
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Update active pairs every 10 steps
        if it % 10 == 0:
            edge_threshold = max(2.0 * r * threshold_factor, 2.0 * best_local_r * threshold_factor)
            active_pairs = get_active_pairs(centers, r, edge_threshold)
            
        grad = np.zeros_like(centers)
        
        # 1. Pairwise violations
        # Violation v_ij = max(0, 2r - d_ij)
        # Gradient of v_ij^2 w.r.t c_i:
        # d(v^2)/dc_i = 2v * dv/dc_i
        # dv/dc_i = - (c_i - c_j) / d_ij  (if d_ij < 2r)
        
        for i, j in active_pairs:
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-8:
                # Coincident, random nudge
                grad[i] += np.random.normal(0, 0.1, 2)
                grad[j] -= np.random.normal(0, 0.1, 2)
                continue
                
            if d < 2 * r:
                v = 2 * r - d
                # grad of v^2 is 2 * v * (c_i - c_j) / d
                grad_vec = 2 * v * (diff / d)
                grad[i] += grad_vec
                grad[j] -= grad_vec
        
        # 2. Wall violations
        # x < -0.5 + r -> v = (-0.5 + r) - x
        # x > 0.5 - r -> v = x - (0.5 - r)
        
        # Left wall
        mask_l = centers[:, 0] < -0.5 + r
        if np.any(mask_l):
            v = (-0.5 + r) - centers[mask_l, 0]
            grad[mask_l, 0] -= 2 * v  # d(v^2)/dx = 2v * dv/dx = 2v * (-1)
            
        # Right wall
        mask_r = centers[:, 0] > 0.5 - r
        if np.any(mask_r):
            v = centers[mask_r, 0] - (0.5 - r)
            grad[mask_r, 0] += 2 * v
            
        # Bottom wall
        mask_b = centers[:, 1] < -0.5 + r
        if np.any(mask_b):
            v = (-0.5 + r) - centers[mask_b, 1]
            grad[mask_b, 1] -= 2 * v
            
        # Top wall
        mask_t = centers[:, 1] > 0.5 - r
        if np.any(mask_t):
            v = centers[mask_t, 1] - (0.5 - r)
            grad[mask_t, 1] += 2 * v
            
        # Gradient step
        centers -= lr * grad
        
        # Project to square boundaries (hard constraint for stability)
        # The optimization handles the radius margin, but we must stay in [-0.5, 0.5]
        # Actually, the wall violations push them in. But if they overshoot, clip.
        # Clipping to [-0.5, 0.5] is safe.
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay learning rate
        lr *= lr_decay
        
        # Check progress
        if it % 50 == 0:
            current_r = _radius(centers)
            if current_r > best_local_r:
                best_local_r = current_r
                
    return centers

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    """
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    x0 = centers.ravel()
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            },
            callback=callback
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
    
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal grid
    hex_centers = _make_hexagonal_grid(N)
    if hex_centers is not None and len(hex_centers) == N:
        candidates.append(hex_centers)
        
    # 2. Square grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    sq_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    candidates.append(sq_centers)
    
    # 3. Random starts
    for i in range(3):
        c_rand = np.random.uniform(-0.4, 0.4, size=(N, 2))
        candidates.append(c_rand)
        
    # Process each candidate
    for idx, centers in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        current_centers = centers.copy()
        
        # Step 1: Graph-Based PGD to expand radius
        # We want to find a configuration that supports a large radius.
        # PGD minimizes violations for a FIXED r.
        # We can try to increase r iteratively.
        
        # Estimate a starting r
        r_init = _radius(current_centers)
        
        # Try to push r up
        # If r_init is small, we can't just set r_target high and expect PGD to work 
        # because the violations will be huge and gradient steps might diverge.
        # However, PGD is a "repair" method. 
        # Strategy: 
        # 1. Use PGD to fix any overlaps for a moderate r (e.g., 0.06).
        # 2. Then use PGD to try to reach a higher r (e.g., 0.071).
        
        targets = [0.060, 0.065, 0.068, 0.070, 0.0713]
        
        for r_t in targets:
            if time.time() - start_time > DEADLINE - 15:
                break
                
            time_left_step = min(5.0, DEADLINE - 15 - (time.time() - start_time))
            if time_left_step <= 0: break
            
            # Run PGD to satisfy constraints for radius r_t
            new_centers = _graph_pgd(current_centers, r_t, time_limit=time_left_step, threshold_factor=1.5)
            
            r_new = _radius(new_centers)
            
            if r_new > _radius(current_centers):
                current_centers = new_centers
            else:
                # If it didn't improve, stick with current
                pass
                
            # If we reached the target, we can stop expanding for this candidate
            # and move to polish.
            if r_new >= r_t - 1e-5:
                break
                
        # Step 2: Polish with L-BFGS-B
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 1.0:
            polished = _local_refine_smooth(current_centers, time_limit=min(8.0, time_left), beta=200.0)
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
                
        # If we have time, do a final high-beta polish on the best so far
        if best_centers is not None and time.time() - start_time < DEADLINE - 2:
            time_left = DEADLINE - 2 - (time.time() - start_time)
            if time_left > 1.0:
                final_polished = _local_refine_smooth(best_centers, time_limit=time_left, beta=500.0)
                r_final = _radius(final_polished)
                
                if r_final > best_r:
                    best_r = r_final
                    best_centers = final_polished.copy()
                    
    if best_centers is None:
        # Fallback
        best_centers = _make_hexagonal_grid(N)
        
    return best_centers

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = _radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r}")
