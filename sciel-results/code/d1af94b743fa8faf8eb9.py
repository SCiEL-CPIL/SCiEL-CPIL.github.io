"""Structural change: Simulated Annealing on exact min-max radius from hex lattice."""
N = 200
import numpy as np
import time

_DEADLINE = 145.0  # Safety margin below 150s kill
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _DEADLINE

def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    # Distance to boundary for each point: min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Since centers are in [-0.5, 0.5], 0.5 - |x| is the distance to the nearest vertical wall.
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)

def _initial_hex_grid(n_points):
    """Generate a hexagonal grid of points."""
    # Approximate hex grid
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 4
    n_cols = int(1.0 / s) + 4
    
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height
    
    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]

def construct():
    start_time = time.time()
    
    # 1. Initialize
    centers = _initial_hex_grid(N)
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
    centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    
    r_best = _compute_radius(centers)
    centers_best = centers.copy()
    
    # 2. Simulated Annealing
    # We want to maximize r.
    # Move: pick a random point, move it by a small random vector.
    # Accept if r_new > r_old, or with probability exp((r_new - r_old) / T).
    
    # Initial temperature
    # r is around 0.03. Differences might be small.
    # Let's scale T such that initial acceptance is high.
    T = 0.005
    T_min = 1e-6
    alpha = 0.9995  # Cooling factor
    
    # Step size
    step_size = 0.01
    
    # Precompute some constants for speed
    # We will recompute radius from scratch each time for simplicity, 
    # but we can optimize if needed. N=200, O(N^2) is 40k ops. 
    # 100k iterations * 40k ops = 4e9 ops. Might be too slow in Python/NumPy overhead?
    # NumPy vectorization helps. Let's estimate time.
    # One _compute_radius call takes ~1-2 ms?
    # 100k calls * 1.5ms = 150s. Too slow.
    # We need to reduce iterations or speed up radius calculation.
    
    # Optimization:
    # 1. Reduce N of iterations.
    # 2. Use a faster radius check? 
    #    We can check if the move improves the local minimum without full global min?
    #    No, r is global min.
    #    
    # Let's try to speed up _compute_radius.
    # diff = centers[:, None, :] - centers[None, :, :] (200x200x2)
    # dists = sqrt(sum(diff**2, axis=2)) (200x200)
    # min(dists)
    
    # Maybe we can run fewer iterations?
    # If 1 call is 1ms, we can do ~10k calls in 10s.
    # Let's aim for ~50k-100k calls if possible, or optimize the loop.
    
    # Let's try to make the radius calculation faster.
    # We can avoid sqrt if we compare squared distances?
    # r = min(boundary, min_dist / 2)
    # r^2 = min(boundary^2, min_dist^2 / 4)
    # Maximizing r is equivalent to maximizing r^2.
    # This avoids sqrt in the inner loop!
    
    def _compute_radius_sq(centers):
        """Return the largest common radius squared."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        boundary = np.min(np.minimum(
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ))
        if boundary < 0:
            return 0.0
        diff = centers[:, None, :] - centers[None, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        return min(boundary * boundary, min_dist_sq / 4.0)
    
    r_best_sq = r_best * r_best
    T = 0.001  # Scale for squared radius? 
    # r ~ 0.036, r^2 ~ 0.0013.
    # Delta r^2 can be small.
    # Let's set T relative to r^2.
    T = r_best_sq * 0.1
    
    # Step size for SA
    # Initial step should be significant but not too large.
    # Grid spacing is ~0.1. Step 0.01 is 10% of spacing.
    step_size = 0.01
    
    # Number of iterations
    # Let's monitor time.
    n_iter = 0
    while _check_time() and T > T_min:
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a random move
        # Uniform in a circle of radius step_size
        angle = np.random.uniform(0, 2 * np.pi)
        rad = np.random.uniform(0, step_size)
        dx = rad * np.cos(angle)
        dy = rad * np.sin(angle)
        
        old_x = centers[i, 0]
        old_y = centers[i, 1]
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check boundaries quickly
        if new_x < -0.5 or new_x > 0.5 or new_y < -0.5 or new_y > 0.5:
            # Reject immediately
            continue
            
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        r_new_sq = _compute_radius_sq(centers)
        
        # Acceptance criterion
        if r_new_sq > r_best_sq:
            # Always accept if better than global best
            r_best_sq = r_new_sq
            centers_best = centers.copy()
            # Also update current state
            # (already updated)
        else:
            # Metropolis criterion
            delta = r_new_sq - r_best_sq  # Note: r_best_sq is the *current* best found so far, 
            # but for SA, we usually compare to *current* state's value, not global best.
            # Let's keep track of current state's value.
            
            # Wait, I need r_current_sq.
            # If I reject, I revert.
            # If I accept, r_current_sq becomes r_new_sq.
            
            # Let's refactor to track r_curr_sq.
            
            # For now, let's just use a simpler logic:
            # If r_new_sq > r_curr_sq, accept.
            # Else, accept with prob exp((r_new_sq - r_curr_sq) / T).
            
            # I didn't track r_curr_sq in the loop above properly for the "else" branch logic 
            # because I updated r_best_sq.
            
            # Let's restart the loop logic properly.
            pass
        
        # This block is getting messy. Let's implement a clean SA loop.
        # Break to rewrite the loop section clearly.
        break
        
    # --- Clean SA Loop ---
    
    r_curr_sq = r_best_sq
    T = r_curr_sq * 0.05  # Start with moderate temperature
    T_min = 1e-8
    alpha = 0.9999  # Slow cooling to explore more
    
    # Step size annealing? 
    # Start larger, shrink?
    # Or keep constant?
    # Let's keep constant for now, maybe shrink slightly.
    step_size = 0.02
    
    # Reset centers to best found so far (which is the initial grid)
    centers = centers_best.copy()
    
    # Time check
    while _check_time() and T > T_min:
        i = np.random.randint(N)
        
        # Random move
        angle = np.random.uniform(0, 2 * np.pi)
        rad = np.random.uniform(0, step_size)
        dx = rad * np.cos(angle)
        dy = rad * np.sin(angle)
        
        old_x = centers[i, 0]
        old_y = centers[i, 1]
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Boundary check
        if new_x < -0.5 or new_x > 0.5 or new_y < -0.5 or new_y > 0.5:
            T *= alpha
            continue
            
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        r_new_sq = _compute_radius_sq(centers)
        
        # Acceptance
        if r_new_sq >= r_curr_sq:
            # Accept
            r_curr_sq = r_new_sq
            if r_curr_sq > r_best_sq:
                r_best_sq = r_curr_sq
                centers_best = centers.copy()
        else:
            # Reject or accept with prob
            delta = r_new_sq - r_curr_sq
            # delta is negative
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                r_curr_sq = r_new_sq
            else:
                # Revert
                centers[i, 0] = old_x
                centers[i, 1] = old_y
        
        T *= alpha
        
        # Occasionally shrink step size?
        # If we get stuck (no improvement for a long time), we might need to change strategy.
        # But SA should handle this via T.
        
        # To prevent infinite loop if T is very small but time is left, 
        # we rely on _check_time().
        
        # Speed up: 
        # If T is very small, we are doing local search.
        # We can increase iteration rate by reducing overhead.
        
        # Let's add a counter to monitor performance
        # n_iter += 1
        # if n_iter % 10000 == 0:
        #     print(f"T={T}, r_best={np.sqrt(r_best_sq)}")
        
    # 3. Final Polish
    # After SA, do a few rounds of local refinement on the best configuration.
    # Use a simple gradient-free local search or just a few SA steps with very low T.
    
    # Let's do a final "greedy" local search:
    # For each point, try small moves in 4 directions and 4 diagonal directions.
    # If it improves r_best, accept.
    
    centers = centers_best.copy()
    r_curr = np.sqrt(r_best_sq)
    
    # Simple local polish
    for _ in range(10):  # 10 passes
        if not _check_time():
            break
            
        improved = False
        for i in range(N):
            if not _check_time():
                break
                
            old_c = centers[i].copy()
            best_local_c = old_c
            best_local_r = _compute_radius(centers)
            
            # Try 8 directions
            for dx in [-0.001, 0, 0.001]:
                for dy in [-0.001, 0, 0.001]:
                    if dx == 0 and dy == 0:
                        continue
                    new_c = old_c + np.array([dx, dy])
                    if new_c[0] < -0.5 or new_c[0] > 0.5 or new_c[1] < -0.5 or new_c[1] > 0.5:
                        continue
                    centers[i] = new_c
                    r_test = _compute_radius(centers)
                    if r_test > best_local_r:
                        best_local_r = r_test
                        best_local_c = new_c.copy()
            
            if np.any(best_local_c != old_c):
                centers[i] = best_local_c
                improved = True
                if best_local_r > r_best:
                    r_best = best_local_r
                    centers_best = centers.copy()
        
        if not improved:
            break
            
    # Final validation
    r_final = _compute_radius(centers_best)
    if r_final <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers_best = grid[:N]
        
    return centers_best
