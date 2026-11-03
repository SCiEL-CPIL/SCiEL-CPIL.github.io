"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _init_hex(N):
    """Initialize points using a hexagonal lattice."""
    # Estimate spacing
    # Area per hex point ~ s^2 * sqrt(3)/2
    # Total area ~ pi
    # N * s^2 * sqrt(3)/2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate a grid large enough
    max_r = 1.5
    k = int(max_r / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append((x, y))
                
    points = np.array(points)
    
    if len(points) >= N:
        # Sort by distance to origin and take closest N
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')
        return points[idx[:N]]
    else:
        # If not enough, pad with random points inside circle
        # This is unlikely with the s estimation, but handle it
        current_N = len(points)
        if current_N == 0:
            return np.random.uniform(-1, 1, (N, 2)) # Fallback
        # Generate random points
        random_points = np.random.uniform(-1, 1, (N - current_N, 2))
        # Ensure inside circle
        dists_r = np.hypot(random_points[:, 0], random_points[:, 1])
        mask = dists_r > 1.0
        random_points[mask] = random_points[mask] / dists_r[mask, np.newaxis] * 0.99
        return np.vstack([points, random_points])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for final checks
    
    # 1. Initialization
    points = _init_hex(N)
    
    # Initial radius estimate
    # Calculate min dist and boundary
    def calc_radius(pts):
        # Boundary
        d_bound = np.hypot(pts[:, 0], pts[:, 1])
        r_bound = 1.0 - np.max(d_bound)
        
        # Pairwise
        # For N=300, full matrix is 90k elements, fast enough for periodic checks
        # But for SA steps, we use incremental.
        # For initialization, do full check
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_bound, r_pair)

    r = calc_radius(points)
    if r <= 0:
        r = 0.01
    
    # SA Parameters
    T = 0.1
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    step_count = 0
    
    # Pre-allocate for speed?
    # We will compute delta cost incrementally.
    
    while time.time() < deadline:
        # 1. Choose a move
        # Random particle
        i = np.random.randint(N)
        
        # Random displacement
        # Step size proportional to T
        step_size = 0.05 * T + 0.001
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_x = points[i, 0] + dx
        new_y = points[i, 1] + dy
        
        # Hard constraint: Must be inside unit circle
        if new_x*new_x + new_y*new_y > 1.0:
            continue # Reject move, don't update time? No, just skip
        
        old_x, old_y = points[i, 0], points[i, 1]
        
        # 2. Calculate Delta Cost
        # Cost = sum_{j!=i} max(0, 2r - d_ij)^2 + max(0, r + |p_i| - 1)^2
        
        # Distances to all other points
        # Vectorized
        dx_all = points[:, 0] - old_x
        dy_all = points[:, 1] - old_y
        d_old = np.hypot(dx_all, dy_all)
        d_old[i] = np.inf # Exclude self
        
        new_dx_all = points[:, 0] - new_x
        new_dy_all = points[:, 1] - new_y
        d_new = np.hypot(new_dx_all, new_dy_all)
        d_new[i] = np.inf
        
        # Boundary terms
        old_r_dist = np.hypot(old_x, old_y)
        new_r_dist = np.hypot(new_x, new_y)
        
        old_boundary_cost = max(0.0, r + old_r_dist - 1.0)**2
        new_boundary_cost = max(0.0, r + new_r_dist - 1.0)**2
        
        # Pairwise costs
        # max(0, 2r - d)^2
        overlap_old = 2.0*r - d_old
        overlap_old[overlap_old < 0] = 0
        cost_old = np.sum(overlap_old**2) + old_boundary_cost
        
        overlap_new = 2.0*r - d_new
        overlap_new[overlap_new < 0] = 0
        cost_new = np.sum(overlap_new**2) + new_boundary_cost
        
        delta_cost = cost_new - cost_old
        
        # 3. Acceptance
        if delta_cost < 0:
            # Accept
            points[i, 0] = new_x
            points[i, 1] = new_y
        else:
            # Metropolis criterion
            if T > T_min:
                prob = np.exp(-delta_cost / T)
                if np.random.rand() < prob:
                    points[i, 0] = new_x
                    points[i, 1] = new_y
        
        # 4. Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Reheat?
            # If we are stuck, we might reheat.
            # Check if we are improving.
        
        step_count += 1
        
        # 5. Adaptive Radius Update
        # Every 1000 steps, check actual radius and adjust r
        if step_count % 1000 == 0:
            # Check time
            if time.time() > deadline - 10.0:
                break
                
            current_r = calc_radius(points)
            if current_r > r:
                # We can expand
                r = current_r * 1.005
                # Reheat slightly to allow expansion
                T = max(T, 0.05)
            else:
                # We are overlapping or too close to boundary
                # Contract r to relieve pressure
                r = current_r * 0.99
                # Cool down more?
                T = max(T, 0.01)
                
    # Final Cleanup
    # Ensure all points are strictly inside
    dists = np.hypot(points[:, 0], points[:, 1])
    mask = dists > 1.0
    if np.any(mask):
        # Scale down the offending points or the whole set?
        # Scaling whole set reduces radius.
        # Projecting points radially to the boundary is safer for validity,
        # but might cause overlaps.
        # Given the hard constraint in SA, this should be rare.
        # If it happens, scale down the specific points slightly.
        points[mask, 0] = points[mask, 0] / dists[mask] * 0.9999
        points[mask, 1] = points[mask, 1] / dists[mask] * 0.9999
        
    return points

# EVOLVE-BLOCK-END
