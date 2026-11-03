"""
Strategy: Many-body force-directed relaxation (Molecular Dynamics).
Instead of fixing only the single worst bottleneck (simulated annealing on min distance),
we treat the circles as particles with short-range repulsive forces (WCA-like) and
long-range attractive forces to the center (to keep them in the box), using velocity
Verlet integration with damping. This allows multiple constraints to be satisfied
simultaneously, escaping the "ratchet" effect of single-bottleneck repair.
"""
import numpy as np
import time
import math

N = 200

def _compute_radius(centers):
    """Compute the max valid radius for the given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distance
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pair distance
    # For N=200, cdist is fast enough (200x200 = 40k ops)
    dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=-1))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _md_relax(centers, time_limit=5.0):
    """
    Molecular Dynamics relaxation.
    Uses a potential energy landscape where overlaps are penalized heavily.
    """
    centers = centers.copy()
    velocities = np.zeros_like(centers)
    
    # Time management
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Parameters
    # We want to maximize the minimum distance.
    # Force model:
    # 1. Repulsion between pairs: If dist < 2*r_target, push apart.
    #    We don't know r_target, so we use a soft repulsion: F ~ 1/d^2 or similar.
    #    Better: Use a WCA-like potential: U(r) = 4 * ((sigma/r)^12 - (sigma/r)^6) + epsilon
    #    But we don't have a fixed sigma. 
    #    Alternative: Simple inverse square repulsion for all pairs, scaled by overlap.
    
    # Let's use a simpler "soft" repulsion that is always active but stronger when close.
    # F_ij = k * (1 - d_ij / d_max)^2 * direction, for d_ij < d_max
    # But we want to find the max d. 
    # Instead, let's use a potential that penalizes closeness: U = sum (1/d^2) for d < cutoff.
    # Minimize U.
    
    # Actually, a better approach for packing:
    # Define a "target" radius r_est based on area.
    # Area per circle = 1/N. r_est = sqrt(1/(N*pi)) * 0.9 (packing factor approx).
    # r_est = sqrt(1/(200*pi)) * 0.95 ≈ 0.038 * 0.95 ≈ 0.036.
    # This is close to the known optimum 0.0366.
    
    r_target = math.sqrt(1.0 / (N * math.pi)) * 0.98
    
    # MD Parameters
    dt = 0.001
    damping = 0.98
    k_repel = 100.0  # Stiffness of repulsion
    k_wall = 50.0    # Stiffness of wall repulsion
    
    # We will run steps until time limit.
    # To avoid numerical instability, we cap forces.
    
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Adaptive step size? Start small.
    
    while time.time() < deadline:
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        # Vectorized calculation of distances and forces
        # d_ij matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-10
        dist = np.sqrt(dist_sq)
        
        # We apply force only if dist < 2 * r_target (overlap region)
        # Force magnitude: k * (2*r_target - dist) / dist * unit_vector?
        # Let's use a linear spring for overlap: F = k * overlap
        # overlap = max(0, 2*r_target - dist)
        
        # To make it smoother and allow "finding" the max radius, 
        # we can use a potential U = k/2 * (2*r_target - dist)^2 for dist < 2*r_target.
        # Force = k * (2*r_target - dist) * (diff / dist)
        
        mask = dist < 2.0 * r_target
        # Zero out diagonal
        np.fill_diagonal(mask, False)
        
        # Calculate force magnitudes
        # overlap = 2*r_target - dist
        # F_vec = k * overlap * (diff / dist)
        # Note: diff[i,j] = c_i - c_j. Force on i is repulsive (away from j), so +diff direction.
        
        overlap = np.zeros_like(dist)
        overlap[mask] = 2.0 * r_target - dist[mask]
        
        # Force magnitude factor
        # F = k * overlap
        # Direction = diff / dist
        # We need to sum forces on i from all j.
        # forces[i] += sum_j ( k * overlap[i,j] * diff[i,j] / dist[i,j] )
        
        # Vectorized sum
        # factor[i,j] = k * overlap[i,j] / dist[i,j]
        factor = np.zeros_like(dist)
        factor[mask] = k_repel * overlap[mask] / dist[mask]
        
        # forces[i] = sum_j factor[i,j] * diff[i,j]
        # diff[i,j] is vector. factor[i,j] is scalar.
        # forces = np.einsum('ij,ijk->ik', factor, diff)
        forces += np.einsum('ij,ijk->ik', factor, diff)
        
        # Wall forces
        # If center is too close to wall, push inward.
        # Walls at -0.5 and 0.5.
        # Safe zone: [-0.5 + r_target, 0.5 - r_target]
        
        # X walls
        # Left wall: x < -0.5 + r_target
        # Force in +x direction
        left_pen = np.maximum(0, (-0.5 + r_target) - centers[:, 0])
        forces[:, 0] += k_wall * left_pen
        
        # Right wall: x > 0.5 - r_target
        right_pen = np.maximum(0, centers[:, 0] - (0.5 - r_target))
        forces[:, 0] -= k_wall * right_pen
        
        # Y walls
        bottom_pen = np.maximum(0, (-0.5 + r_target) - centers[:, 1])
        forces[:, 1] += k_wall * bottom_pen
        
        top_pen = np.maximum(0, centers[:, 1] - (0.5 - r_target))
        forces[:, 1] -= k_wall * top_pen
        
        # Update velocities (Velocity Verlet-like, simple Euler with damping)
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # Project to square (hard constraint)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Check improvement
        # Do this every few steps to save time
        if int((time.time() - start_time) * 100) % 10 == 0:
             current_r = _compute_radius(centers)
             if current_r > best_r:
                 best_r = current_r
                 best_centers = centers.copy()
        
        # Adaptive r_target: 
        # If we are stuck, maybe r_target is too high? 
        # Or we can slowly increase r_target to force denser packing.
        # Let's try a slow increase of r_target to push the system to higher density.
        # But if r_target > current_r, the system will be in high energy state, 
        # but MD should find a local min.
        # Let's stick to fixed r_target for stability, but maybe increase it slightly over time.
        # r_target = r_target * 1.00001
        
    return best_centers, best_r

def _generate_hex_initial(rows):
    """
    Generate a hexagonal packing layout.
    """
    cols = math.ceil(N / rows)
    
    d_w = 1.0 / (cols - 0.5)
    d_h = 1.0 / ((rows - 1) * math.sqrt(3)/2 + 1.0)
    d = min(d_w, d_h)
    
    pts = []
    row_spacing = d * math.sqrt(3) / 2
    
    for i in range(rows):
        y = -0.5 + d/2 + i * row_spacing
        if i % 2 == 0:
            n_in_row = cols
            x_start = -0.5 + d/2
        else:
            n_in_row = cols - 1
            x_start = -0.5 + d
            
        for j in range(n_in_row):
            x = x_start + j * d
            pts.append([x, y])
            
    pts = np.array(pts)
    
    if len(pts) < N:
        return None
    if len(pts) > N:
        pts = pts[:N]
        
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration, ensuring no immediate overlaps if possible.
    """
    centers = np.random.uniform(-0.5, 0.5, (N, 2))
    return centers

def construct():
    start_time = time.time()
    # Safety margin: 15s
    deadline = start_time + 135.0
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate multiple diverse initializations
    # Hex is usually a good start.
    # Random can help escape bad hex local minima.
    
    initializations = []
    
    # Try different hex rows
    for rows in [13, 14, 15, 16]:
        pts = _generate_hex_initial(rows)
        if pts is not None:
            initializations.append(pts)
            
    # Add a few random starts
    for _ in range(3):
        initializations.append(_generate_random_initial())
        
    # 2. Run MD relaxation on each
    # Allocate time based on number of initializations
    # We have ~135s. 7 initializations. ~19s each.
    # But we want to spend more time on the best ones?
    # Let's run all with a fixed time budget, then refine the best.
    
    time_per_init = 15.0
    
    for init_pts in initializations:
        if time.time() > deadline - 10:
            break
            
        # Check if we have enough time
        time_left = deadline - time.time() - 5
        if time_left < 5:
            break
            
        current_limit = min(time_per_init, time_left)
        
        optimized, r_opt = _md_relax(init_pts, time_limit=current_limit)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
            
    # 3. Final refinement on the best solution found
    if best_centers is not None and time.time() < deadline - 5:
        time_left = deadline - time.time() - 2
        if time_left > 10:
            # Run MD again on the best, maybe with a slightly higher r_target?
            # Or just more time.
            # Let's just run more MD.
            optimized, r_opt = _md_relax(best_centers, time_limit=time_left)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()

    # Fallback if something went wrong
    if best_centers is None:
        # Grid fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
