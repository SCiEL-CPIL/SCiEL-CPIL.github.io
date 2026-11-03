import numpy as np
import time

N = 300
TIME_LIMIT = 145.0 # Safety margin from 150s limit

def _init_hex():
    """
    Generate a dense hexagonal lattice, clip to unit circle,
    and select the N points closest to the center.
    """
    # Target radius is approx 0.053. Spacing should be slightly larger than 2*r ~ 0.106
    # Let's start with a tighter lattice to ensure density, then let dynamics expand it.
    spacing = 0.09
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    
    mask = r < 0.99
    points = points[mask]
    r = r[mask]
    
    idx = np.argsort(r)
    points = points[idx]
    
    if len(points) < N:
        # Fallback if lattice is too sparse (should not happen with k~14)
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
    else:
        points = points[:N]
        
    return points

def _compute_radius(points):
    """
    Compute the actual packing radius: min(pairwise_dist/2, boundary_dist)
    """
    # Pairwise distances
    # For N=300, 300x300 matrix is fine in memory (90k floats)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distance
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary)

def _simulate(points_init, time_budget):
    """
    Rigid-body collision dynamics simulation using Hertzian contact law.
    Uses Verlet integration.
    Force F ~ (overlap)^4 (Hertzian for 2D disks is actually (overlap)^(3/2) for energy, 
    but force is (overlap)^(1/2). However, the prompt specifies "force proportional to 
    overlap depth to the fourth power". This is a stiff potential, good for hard contacts.
    Let's follow the prompt: F = k * (delta)^4 * direction.
    Actually, standard Hertzian contact in 2D: F ~ delta^(1/2). 
    But the prompt says: "repulsive force proportional to the overlap depth to the fourth power".
    I will implement F = k * (delta)^4. This creates a very stiff repulsion.
    """
    
    current_points = points_init.copy()
    velocities = np.zeros_like(current_points)
    
    best_points = current_points.copy()
    best_score = _compute_radius(current_points)
    
    # Parameters
    # Target radius estimate to define "overlap"
    # We don't know the optimal r exactly, but we can estimate it.
    # Or we can define overlap relative to a dynamic threshold.
    # However, the prompt implies a specific law.
    # Let's define overlap = max(0, 2*r_target - dist).
    # If we use a fixed r_target, we are optimizing for that r.
    # Better: use a soft potential that pushes all points apart, weighted by closeness.
    # But the prompt specifically asks for "Hertzian contact law... force proportional to overlap depth to the fourth power".
    # This usually implies a hard constraint at r_opt.
    # Since we don't know r_opt, we can run the simulation with a slightly smaller r_target,
    # or use a potential that acts as a barrier.
    
    # Alternative interpretation: The "overlap" is relative to the *current* minimal distance?
    # No, that's unstable.
    
    # Let's use a potential energy function that approximates the packing problem.
    # V(d) = 0 if d >= 2r, else k * (2r - d)^4.
    # We need to choose r. Let's start with a rough estimate of the optimal radius.
    # Area argument: 300 * pi * r^2 <= pi * 1^2 => r <= 1/sqrt(300) ~ 0.0577.
    # Best known is 0.05315.
    # Let's set r_target = 0.053.
    
    r_target = 0.053
    k_force = 100.0  # Stiffness
    dt = 0.01        # Time step
    damping = 0.99   # Velocity damping to help convergence
    
    start_sim = time.time()
    
    # Pre-allocate arrays for speed
    # We will compute forces in a loop over pairs? No, vectorized is better.
    # But N=300, 45k pairs. Vectorized force calculation is fast.
    
    while time.time() - start_sim < time_budget:
        # Compute forces
        forces = np.zeros_like(current_points)
        
        # Pairwise distances
        diffs = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap: max(0, 2*r_target - dist)
        overlaps = np.maximum(0.0, 2.0 * r_target - dists)
        
        # Force magnitude: k * overlap^4
        # Only non-zero where overlap > 0
        # F_ij = k * (overlap)^4 * (p_i - p_j) / dist
        # Note: The force on i is in the direction away from j.
        # Vector from j to i is (p_i - p_j).
        
        # To avoid division by zero and to only compute where needed:
        # We can mask the calculation.
        mask = overlaps > 1e-9
        if np.any(mask):
            # Get indices where overlap exists
            i_idx, j_idx = np.where(mask)
            # We need to sum forces on each i from all j.
            # This is tricky to do in a single vectorized op without scatter-add.
            # Since N is small (300), a loop over i might be acceptable if optimized, 
            # or we can use bincount.
            
            # Let's use bincount for efficiency.
            # Force vector on i from j: F_vec = k * overlap^4 * (p_i - p_j) / dist
            # This is not a scalar to bincount directly.
            
            # Alternative: Loop over pairs is 45k iterations. In Python, this is slow (~100ms+).
            # Vectorized approach:
            # Compute full force matrix F_matrix[i,j] = scalar_factor * unit_vec[i,j]
            # Then sum over j for each i.
            
            # Unit vectors
            # dists is inf on diagonal, so division is safe for non-diagonal.
            # But we only care about mask.
            # Let's compute unit vectors for all pairs (safe because dists>0 off-diag)
            unit_vecs = diffs / dists[..., np.newaxis]
            
            # Force scalar magnitude for each pair
            # f_mag = k_force * overlaps^4
            f_mag = k_force * (overlaps ** 4)
            
            # Force vector matrix
            # F_matrix[i,j] = f_mag[i,j] * unit_vecs[i,j]
            # We want Force_on_i = sum_j F_matrix[i,j]
            # Note: F_matrix[i,j] is the force exerted BY j ON i?
            # diffs = p_i - p_j. So unit_vecs points from j to i.
            # If overlap > 0, they are too close. We want to push i away from j.
            # So direction is correct.
            
            F_matrix = f_mag[..., np.newaxis] * unit_vecs
            
            # Sum over j (axis=1) to get net force on each i
            forces = np.sum(F_matrix, axis=1)
            
            # Boundary forces
            # If a point is outside the "safe" zone (radius > 1 - r_target), push it in.
            radii = np.linalg.norm(current_points, axis=1)
            boundary_overlap = np.maximum(0.0, radii - (1.0 - r_target))
            
            # Force direction: towards center. -p_i / |p_i|
            # Magnitude: k_b * boundary_overlap^4
            # Use same stiffness or different?
            k_boundary = 100.0
            b_f_mag = k_boundary * (boundary_overlap ** 4)
            
            # Unit vector towards center
            # p_i / |p_i|
            safe_radii = np.where(radii > 1e-6, radii, 1.0)
            unit_radial = current_points / safe_radii[:, np.newaxis]
            
            # Force is -unit_radial * b_f_mag
            boundary_forces = -b_f_mag[:, np.newaxis] * unit_radial
            
            forces += boundary_forces
        
        # Verlet Integration
        # v_new = (v_old + a * dt) * damping
        # x_new = x_old + v_new * dt
        
        accelerations = forces # mass = 1
        velocities = (velocities + accelerations * dt) * damping
        current_points = current_points + velocities * dt
        
        # Check if any point is way outside, clamp if necessary to prevent explosion
        # The boundary force should handle it, but just in case:
        radii = np.linalg.norm(current_points, axis=1)
        if np.any(radii > 1.1):
            # Hard clamp to 1.0
            scale = np.where(radii > 1.0, 1.0 / radii, 1.0)
            current_points = current_points * scale[:, np.newaxis]
            velocities = velocities * 0.5 # Damp velocity if clamped
        
        # Evaluate score
        # To save time, we can evaluate score every few steps, but N=300 is fast enough
        current_score = _compute_radius(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
            # If we reach a very high score, we might want to tighten r_target?
            # Or just let it settle.
            
        # Adaptive dt?
        # If score is stable for a while, we can reduce dt to find finer local optimum.
        # But for now, fixed dt with damping should converge.
        
        # Check time
        if time.time() - start_sim > time_budget:
            break
            
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Multi-start strategy
    # We have ~145s.
    # Let's do 10 restarts with ~14s each.
    # Or fewer restarts with longer runs?
    # Dynamics can get stuck in local minima. More restarts is better.
    
    best_points = None
    best_score = 0.0
    
    num_restarts = 10
    time_per_restart = (TIME_LIMIT - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Generate initial points
        # Mix of hex and random to explore different basins
        if i % 2 == 0:
            init_points = _init_hex()
        else:
            # Random uniform in circle, slightly perturbed hex?
            # Let's just use random for diversity
            theta = np.random.uniform(0, 2 * np.pi, N)
            rho = np.sqrt(np.random.uniform(0, 0.95, N))
            init_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
        # Run simulation
        local_points, local_score = _simulate(init_points, time_per_restart)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_points
            
        # If we have a very good score, we can spend less time on subsequent random restarts
        # or break early if we hit a theoretical bound? No known bound.
        
    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _init_hex()
    
    # Final validation and cleanup
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        # Fallback to a safe configuration
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
    return best_points
