"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    # Distance between all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls: center must be at least r away from each wall
    # Square is [-0.5, 0.5]^2
    min_wall = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    r = min(min_pair_dist / 2.0, min_wall)
    return r


def _initial_hex(n):
    """Create a hexagonal lattice initialization for n circles in the unit square."""
    # Hexagonal packing: rows offset
    # We want to fit n circles. Let's try different row counts.
    # For hexagonal packing with radius r, horizontal spacing is 2r, vertical is sqrt(3)*r.
    # But we're placing centers, so let's just create a dense hex pattern and trim.
    
    # Try to find a good arrangement
    # Let's use a grid that approximates hexagonal
    # For 50 circles, maybe 8 rows x 7 cols with some adjustment?
    # Or 7 rows x 8 cols?
    
    # Let's create a hexagonal lattice and pick the best n points
    # Actually, let's just create a square grid first and then optimize
    # But the directive says hexagonal is better. Let's do a proper hex init.
    
    # Hexagonal lattice:
    # Row i has points at y = i * dy
    # Even rows: x = j * dx
    # Odd rows: x = (j + 0.5) * dx
    # dx = 2*r_init, dy = sqrt(3)*r_init
    # But we don't know r_init. Let's just space them out.
    
    # For 50 circles in a unit square, let's try:
    # 8 rows, alternating 7 and 6 points? 8*7=56, 8*6=48. 
    # Let's do 7 rows: 8,7,8,7,8,7,8 = 53 points, trim to 50.
    # Or 6 rows: 9,8,9,8,9,8 = 51, trim to 50.
    
    # Let's use a simpler approach: create a hex-like pattern
    # and center it in the square.
    
    rows = 8
    # Points per row: alternate between 7 and 6
    points_per_row = [7, 6, 7, 6, 7, 6, 7, 6]  # total = 52, trim to 50
    # Actually let's do: 7,7,7,7,7,7,7,5 = 52, trim to 50
    # Or just use a regular grid for simplicity and let SA do the work.
    
    # Let me just use a dense square grid as initial and let SA optimize.
    # The hex init is mentioned as better, so let's do it properly.
    
    # Create hexagonal lattice points
    dx = 0.15  # horizontal spacing
    dy = dx * np.sqrt(3) / 2  # vertical spacing for hex
    
    # Generate points
    points = []
    for i in range(15):
        y = i * dy - 0.5
        for j in range(15):
            x = j * dx - 0.5 + (0.5 * dx if i % 2 else 0)
            points.append([x, y])
    
    points = np.array(points)
    # Filter to those inside the square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than n, randomly sample n
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        extra = np.random.uniform(-0.5, 0.5, (n - len(points), 2))
        points = np.vstack([points, extra])
    
    return points


def _simulated_annealing(centers, max_iter=200000, T_start=0.1, T_end=1e-6, cooling=0.9999):
    """Simulated annealing to optimize circle packing."""
    n = len(centers)
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    current_r = best_r
    T = T_start
    
    for it in range(max_iter):
        # Pick a random circle to move
        i = np.random.randint(n)
        # Random displacement
        scale = T * 10  # scale the perturbation with temperature
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        new_x = centers[i, 0] + dx
        new_y = centers[i, 1] + dy
        
        # Clamp to square
        new_x = np.clip(new_x, -0.5, 0.5)
        new_y = np.clip(new_y, -0.5, 0.5)
        
        old_x, old_y = centers[i, 0], centers[i, 1]
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        new_r = _compute_radius(centers)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() > prob:
                    # Reject
                    centers[i, 0] = old_x
                    centers[i, 1] = old_y
            else:
                current_r = new_r
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    return best_centers, best_r


def _local_refinement(centers, max_iter=50000):
    """Local refinement using gradient-free optimization."""
    n = len(centers)
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    step_size = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for dx in [0, -step_size, step_size]:
                for dy in [0, -step_size, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    old_x, old_y = centers[i, 0], centers[i, 1]
                    new_x = np.clip(old_x + dx, -0.5, 0.5)
                    new_y = np.clip(old_y + dy, -0.5, 0.5)
                    centers[i, 0] = new_x
                    centers[i, 1] = new_y
                    new_r = _compute_radius(centers)
                    if new_r > current_r:
                        current_r = new_r
                        improved = True
                    else:
                        centers[i, 0] = old_x
                        centers[i, 1] = old_y
        if not improved:
            step_size *= 0.5
            no_improve += 1
            if no_improve > 10 or step_size < 1e-6:
                break
        else:
            no_improve = 0
    
    return centers, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    # Initial hexagonal configuration
    centers = _initial_hex(N)
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Simulated annealing
    remaining_iters = 200000
    sa_centers, sa_r = _simulated_annealing(
        centers, 
        max_iter=remaining_iters,
        T_start=0.1,
        T_end=1e-6,
        cooling=0.99995
    )
    
    if sa_r > best_r:
        best_r = sa_r
        best_centers = sa_centers
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_centers
    
    # Local refinement
    ref_centers, ref_r = _local_refinement(best_centers, max_iter=20000)
    
    if ref_r > best_r:
        best_r = ref_r
        best_centers = ref_centers
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # One more SA pass with lower temperature
        sa2_centers, sa2_r = _simulated_annealing(
            best_centers,
            max_iter=50000,
            T_start=0.01,
            T_end=1e-7,
            cooling=0.9999
        )
        if sa2_r > best_r:
            best_r = sa2_r
            best_centers = sa2_centers
    
    return best_centers
# EVOLVE-BLOCK-END
