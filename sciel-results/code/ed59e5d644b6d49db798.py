"""Pack 50 equal circles in a unit square using 2D Elastic System (Symplectic Integrator)."""
import numpy as np
import time

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

def _elastic_simulate(centers, time_limit=5.0, k_spring=100.0, k_wall=500.0, dt=0.001, damping=0.99):
    """
    Simulate a 2D elastic system where circles are connected by stiff springs
    when their distance falls below a dynamic threshold.
    Use a symplectic integrator (Verlet-like) with adaptive damping to resolve overlaps.
    """
    centers = centers.copy()
    velocities = np.zeros_like(centers)
    start_time = time.time()
    
    # Adaptive damping: start high to settle quickly, then reduce to allow fine tuning
    # We'll just use a fixed high damping for stability, as we are not looking for oscillations
    # but for a stable equilibrium.
    
    # Threshold for spring activation
    # We want to push apart circles that are close.
    # The "rest length" is 0, but we only activate when distance < threshold.
    # Let's set threshold dynamically based on current min distance or a fixed value.
    # A fixed threshold of ~0.2 (larger than expected diameter 0.15) should be safe.
    threshold = 0.2
    
    for it in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise spring forces
        # We only compute forces for pairs within threshold to save time (though N=50 is small)
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < threshold * threshold:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-8:
                        # Coincident, random nudge
                        forces[i] += np.random.normal(0, 0.1, 2)
                        forces[j] -= np.random.normal(0, 0.1, 2)
                    else:
                        # Spring force: F = -k * (dist - rest) * dir
                        # Here rest = 0, so we want to push them apart.
                        # F = k * (threshold - dist) * dir  (if dist < threshold)
                        # Actually, to maximize minimum distance, we want to push them apart
                        # proportional to how close they are.
                        # Let's use F = k * (threshold - dist) / threshold * dir
                        force_mag = k_spring * (threshold - dist) / threshold
                        force_vec = force_mag * (diff / dist)
                        forces[i] += force_vec
                        forces[j] -= force_vec
        
        # Wall forces
        # Push circles away from walls
        # Left wall
        mask_l = centers[:, 0] < -0.5 + 0.05  # Use a margin
        if np.any(mask_l):
            # Force proportional to penetration
            pen = (-0.5 + 0.05) - centers[mask_l, 0]
            forces[mask_l, 0] += k_wall * pen
            
        # Right wall
        mask_r = centers[:, 0] > 0.5 - 0.05
        if np.any(mask_r):
            pen = centers[mask_r, 0] - (0.5 - 0.05)
            forces[mask_r, 0] -= k_wall * pen
            
        # Bottom wall
        mask_b = centers[:, 1] < -0.5 + 0.05
        if np.any(mask_b):
            pen = (-0.5 + 0.05) - centers[mask_b, 1]
            forces[mask_b, 1] += k_wall * pen
            
        # Top wall
        mask_t = centers[:, 1] > 0.5 - 0.05
        if np.any(mask_t):
            pen = centers[mask_t, 1] - (0.5 - 0.05)
            forces[mask_t, 1] -= k_wall * pen
        
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Update positions
        centers += velocities * dt
        
        # Hard constraint: clip to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # If we hit a wall, zero out the velocity component to prevent sticking
        centers[(centers[:, 0] == -0.5) | (centers[:, 0] == 0.5), 0] = centers[(centers[:, 0] == -0.5) | (centers[:, 0] == 0.5), 0]
        # This is a bit messy, let's just let the force handle it.
        
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
    from scipy.optimize import minimize
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
        
        # Step 1: Elastic simulation to resolve overlaps and expand
        # We run the simulation for a fixed time, then polish.
        time_left_step = min(5.0, DEADLINE - 15 - (time.time() - start_time))
        if time_left_step > 0:
            new_centers = _elastic_simulate(current_centers, time_limit=time_left_step)
            r_new = _radius(new_centers)
            
            if r_new > _radius(current_centers):
                current_centers = new_centers
            else:
                # If it didn't improve, stick with current
                pass
                
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
