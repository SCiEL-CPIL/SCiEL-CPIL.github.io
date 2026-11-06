"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a packing of N equal circles in a unit square [-0.5, 0.5]^2.
    Strategy: Simulated Annealing with vectorized repulsive forces.
    We minimize a potential energy that penalizes overlaps (distance < 2r) 
    and pushes points to boundaries if they are too close to the edge.
    To maximize radius, we run SA to find a dense configuration, 
    then compute the maximum possible radius for that configuration.
    However, a more direct approach for SA is to optimize the positions 
    to maximize the minimum distance to neighbors and boundaries.
    
    Since maximizing min distance is non-smooth, we use a soft potential 
    that approximates the "hard sphere" repulsion.
    We will perform SA to minimize the sum of squared overlaps for a 
    fixed target radius, but since we want to find the *best* radius, 
    we can instead minimize a potential that favors larger separations.
    
    Actually, a standard trick is to minimize the "energy" defined as:
    E = sum_{i<j} max(0, 1 - d_ij)^2 + sum_i max(0, 1 - d_i_boundary)^2
    where d_ij is normalized distance. But this is tricky to scale.
    
    Better Strategy: 
    1. Start with a hexagonal grid initialization (good packing density).
    2. Use Simulated Annealing to minimize the "overlap energy" for a 
       slightly larger radius than the grid allows, effectively pushing 
       points apart until they settle into a local minimum of the 
       hard-core repulsion.
    3. The score is determined by the minimum distance in the final config.
    
    To be robust and fast:
    - Use vectorized numpy operations for force calculation.
    - Time limit: 140 seconds.
    - Return the best configuration found.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal packing
    # For N=200, a roughly 14x15 grid is good.
    # Hex packing has density ~0.9069.
    # Area per circle ~ pi*r^2. Total area 1.
    # 200 * pi * r^2 <= 1 => r <= sqrt(1/(200*pi)) ~ 0.0399.
    # Current best is 0.0366.
    
    # Generate hexagonal grid points
    # Rows of alternating length
    # Let's just generate a dense hex grid and take the first N points
    # centered at origin.
    
    points = []
    # Hex lattice vectors
    # a = 1 (spacing)
    # We will scale later.
    # Generate points in a large enough area
    for row in range(-20, 21):
        for col in range(-20, 21):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            points.append((x, y))
    
    points = np.array(points)
    # Normalize to fit in [-0.5, 0.5] roughly
    # The extent of 40x40 hex grid is approx 40 * 1.5 = 60 in x, 40*0.866=34 in y
    # We want to map this to [-0.5, 0.5]
    # But we only need N=200 points.
    # Let's select the N closest points to the origin from a generated hex grid.
    
    # Generate more points to ensure we have >200 near center
    pts = []
    for row in range(-10, 11):
        for col in range(-10, 11):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    
    # Distance to origin
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:N]]
    
    # Scale pts to fit in [-0.5, 0.5]
    # Max extent in hex grid:
    # The N=200 points will cover a certain area.
    # Let's just normalize so that the max absolute value is 0.5
    max_abs = np.max(np.abs(pts))
    pts = pts / max_abs * 0.49  # Keep slightly inside
    
    # Ensure pts is float
    pts = pts.astype(np.float64)
    
    # 2. Simulated Annealing
    # Objective: Minimize "Overlap Energy"
    # E = sum_{i<j} (1/d_ij - 1/r_target)^2 if d_ij < r_target else 0
    # This is hard to tune r_target.
    
    # Alternative: Minimize sum of inverse distances? No, that favors clustering.
    # We want to MAXIMIZE the minimum distance.
    # This is equivalent to minimizing the maximum overlap.
    
    # Let's use a soft potential:
    # V(r) = 1/r^12 - 1/r^6 (Lennard-Jones) scaled appropriately.
    # Or simpler: V(r) = (1/r)^k for r < 1, 0 otherwise.
    # If we minimize sum V(d_ij), points will repel each other.
    # We also need to keep them in the box.
    # V_boundary = (1/d_boundary)^k if d_boundary < margin else 0.
    
    # Let's use a force-based SA.
    # Force on i from j: F_ij = k / d_ij^2 * (r_ij / d_ij) if d_ij < R_repel else 0
    # We want to push them apart.
    
    # Let's try minimizing:
    # E = sum_{i<j} max(0, 1 - d_ij)^2 + sum_i max(0, 1 - d_i_bound)^2
    # This is a "soft" hard sphere potential.
    # We want d_ij >= 1 (in normalized units where min dist is 1).
    # But we don't know the scale.
    
    # Let's stick to the actual coordinates.
    # We want to maximize r_min = min( min_{i<j} d_ij/2, min_i min_k (0.5 - |c_ik|) )
    
    # Let's define a potential that penalizes distances below a threshold D.
    # If we set D=1, and coordinates are in [-0.5, 0.5], this is too large.
    # Let's work in a scaled space where the box is [-1, 1].
    # Then r_max ~ 0.073.
    
    # Let's rescale points to [-1, 1]
    pts = pts * 2.0
    
    # Potential:
    # E = sum_{i<j} (1/d_ij - 1)^2 * w if d_ij < 1 else 0
    # Plus boundary penalty:
    # E += sum_i (1/|x_i| - 1)^2 * w if |x_i| < 1 else 0 (for x)
    # E += sum_i (1/|y_i| - 1)^2 * w if |y_i| < 1 else 0 (for y)
    
    # This potential encourages d_ij >= 1 and |coord| >= 1? No, |coord| <= 1.
    # Boundary penalty should push points INSIDE if they are too close to edge?
    # No, we want them INSIDE. The constraint is |coord| <= 1.
    # The radius is limited by distance to boundary.
    # So we want to maximize min(d_ij/2, 1 - max(|x_i|, |y_i|)).
    
    # Let's use a simple repulsive potential that is always active but decays.
    # V(d) = 1/d^4. Minimize sum V(d_ij).
    # This will push points apart.
    # We also need a confining potential to keep them in the box.
    # V_box(x) = x^4 for |x| > 0.9? No, that pushes them out.
    # We need a potential that penalizes going OUTSIDE the box.
    # V_box(x) = 0 if |x| <= 1, (|x|-1)^4 if |x| > 1.
    
    # So, Total Energy E = sum_{i<j} 1/d_ij^4 + sum_i [V_box(x_i) + V_box(y_i)]
    
    # If we minimize this, points will spread out until they hit the box walls.
    # The equilibrium will be a packing where the repulsive forces balance the box walls.
    # This should approximate the optimal packing for maximizing the minimum distance,
    # assuming the repulsive force is strong enough to prevent overlaps.
    
    # Let's refine:
    # We want to maximize r.
    # If we use 1/d^4, small distances are heavily penalized.
    # This is good.
    
    def compute_forces(pts):
        """
        Computes forces on all points.
        pts: (N, 2) array.
        Returns: (N, 2) array of forces.
        """
        N = len(pts)
        forces = np.zeros_like(pts)
        
        # Pairwise distances
        # diff[i, j] = pts[i] - pts[j]
        # To save memory, we can do this in chunks or use broadcasting if N is small.
        # N=200, so 200x200 matrix is fine (40k entries).
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: F = -dV/dd = -d(1/d^4)/dd = 4/d^5
        # Direction: diff / d
        # So F = 4 * diff / d^6
        # d^6 = (d^2)^3 = dist_sq^3
        
        # We only want repulsive force.
        # F_vec = 4 * diff / (dist_sq^3 + eps)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Compute forces
        # dist_sq is (N, N)
        # diff is (N, N, 2)
        
        # We want to sum forces from all j on i.
        # F_i = sum_j 4 * (pts[i] - pts[j]) / |pts[i] - pts[j]|^6
        
        # Let's compute the coefficient
        # coeff = 4 / dist_sq^3
        # coeff[i, i] = 0
        
        dist_sq_safe = dist_sq.copy()
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        coeff = 4.0 / (dist_sq_safe ** 3)
        np.fill_diagonal(coeff, 0.0)
        
        # F = coeff[..., np.newaxis] * diff
        # F is (N, N, 2)
        # Sum over j (axis 1)
        forces_pair = np.sum(coeff[..., np.newaxis] * diff, axis=1)
        
        # Boundary forces
        # V_box(x) = (|x|-1)^4 for |x|>1
        # dV/dx = 4(|x|-1)^3 * sign(x)
        # Force = -dV/dx
        # So if x > 1, F_x = -4(x-1)^3
        # If x < -1, F_x = 4(x+1)^3 (since sign is -1, -dV/dx = -4(x+1)^3*(-1) = 4(x+1)^3)
        
        # Let's use a softer potential for boundary to avoid spikes.
        # Or just clamp positions after update.
        # Clamping is easier and more stable for SA.
        
        return forces_pair
    
    # SA Parameters
    T = 0.1  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 0.001
    steps_per_temp = 10
    
    # Best score tracker
    def compute_score(pts):
        # pts in [-1, 1]
        # Radius is min( min_{i<j} d_ij/2, min_i (1 - max(|x_i|, |y_i|)) )
        # Note: max(|x|, |y|) is distance to closest vertical/horizontal wall?
        # No, distance to boundary of square [-1,1]^2 is 1 - max(|x|, |y|).
        # Correct.
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        min_dist = np.sqrt(np.min(dist_sq))
        
        # Boundary distances
        max_abs = np.max(np.abs(pts), axis=1)
        min_bound = 1.0 - np.max(max_abs)
        
        r = min(min_dist / 2.0, min_bound)
        return r
    
    best_pts = pts.copy()
    best_score = compute_score(best_pts)
    
    # SA Loop
    while T > T_min:
        for _ in range(steps_per_temp):
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Generate random perturbation
            # Gaussian noise
            noise = np.random.normal(0, np.sqrt(T) * 0.1, size=pts.shape)
            new_pts = pts + noise
            
            # Compute energy difference
            # E = sum 1/d^4 + boundary penalty
            # For speed, we can approximate or just use the score function?
            # Score function is expensive (O(N^2)).
            # Let's use a simplified energy for the SA step.
            
            # Actually, let's just use the force-based update?
            # No, SA needs energy.
            
            # Let's compute the "overlap" energy quickly.
            # E = sum_{i<j} max(0, 1 - d_ij)^2
            # This is O(N^2).
            
            # To save time, we can update forces and use gradient descent steps
            # mixed with random kicks?
            # Let's stick to pure SA with a fast energy estimate.
            
            # Fast Energy:
            # diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
            # dist_sq = np.sum(diff**2, axis=-1)
            # np.fill_diagonal(dist_sq, 1.0)
            # d = np.sqrt(dist_sq)
            # overlap = np.maximum(0, 1.0 - d)
            # E = np.sum(overlap**2)
            
            # This is still O(N^2) but vectorized. N=200 is fast.
            
            # Let's compute E for old and new
            # We already have pts.
            
            # Old Energy
            diff_old = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq_old = np.sum(diff_old**2, axis=-1)
            np.fill_diagonal(dist_sq_old, 1.0)
            d_old = np.sqrt(dist_sq_old)
            overlap_old = np.maximum(0, 1.0 - d_old)
            E_old = np.sum(overlap_old**2)
            
            # New Energy
            diff_new = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
            dist_sq_new = np.sum(diff_new**2, axis=-1)
            np.fill_diagonal(dist_sq_new, 1.0)
            d_new = np.sqrt(dist_sq_new)
            overlap_new = np.maximum(0, 1.0 - d_new)
            E_new = np.sum(overlap_new**2)
            
            # Boundary Penalty
            # E_bound = sum max(0, max(|x|,|y|) - 1)^2
            max_abs_old = np.max(np.abs(pts), axis=1)
            E_bound_old = np.sum(np.maximum(0, max_abs_old - 1.0)**2)
            
            max_abs_new = np.max(np.abs(new_pts), axis=1)
            E_bound_new = np.sum(np.maximum(0, max_abs_new - 1.0)**2)
            
            E_old_total = E_old + E_bound_old
            E_new_total = E_new + E_bound_new
            
            dE = E_new_total - E_old_total
            
            # Accept if dE < 0 or with probability exp(-dE/T)
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                pts = new_pts
                # Clamp to box to prevent escape
                pts = np.clip(pts, -1.0, 1.0)
                
                # Check if this is a better score
                # We don't want to compute full score every step (expensive).
                # But we can do it every 100 steps.
                
        # Cool down
        T *= alpha
        
        # Check best score every temperature step
        if time.time() > deadline - 10.0:
            break
            
        current_score = compute_score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
    
    # Final refinement: Local optimization (Gradient Descent)
    # Use the best point found by SA as a starting point for a short GD run.
    pts = best_pts.copy()
    
    # GD for 1000 steps
    lr = 0.01
    for i in range(1000):
        if time.time() > deadline - 2.0:
            break
        
        forces = compute_forces(pts)
        
        # Update
        pts = pts - lr * forces
        
        # Clamp
        pts = np.clip(pts, -1.0, 1.0)
        
        # Check score every 100 steps
        if i % 100 == 0:
            current_score = compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
    
    # Rescale back to [-0.5, 0.5]
    best_pts = best_pts / 2.0
    
    # Final validation: ensure all points are strictly inside
    # The score calculation handles the boundary.
    # But we must ensure no coincident points or NaNs.
    
    # Return
    return best_pts

# EVOLVE-BLOCK-END
