import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        for _ in range(20):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.98
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.uniform(0, 0.9, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    pts = hex_lattice_init(N)
    velocities = np.zeros_like(pts)
    
    def compute_radius(pts):
        # Efficient min pair distance
        from scipy.spatial.distance import pdist
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()

    # MD Parameters
    dt = 0.001
    k_rep = 100.0  # Repulsion strength
    k_wall = 100.0 # Wall strength
    damping = 0.99 # Velocity damping per step
    target_d = 0.0 # We will effectively maximize the min distance by repulsion

    # We simulate for a fixed number of steps or until time limit
    # Each step computes O(N^2) forces. N=300 -> 90k pairs. 
    # 90k * 1000 steps = 90M ops. In numpy this is fast.
    # Let's do ~5000 steps total, split into annealing phases.

    steps_per_phase = 1000
    num_phases = 5
    total_steps = steps_per_phase * num_phases

    for phase in range(num_phases):
        # Anneal damping: start with more damping (cooler), then less? 
        # Actually, for packing, we usually want to "shake" to escape local minima, then cool.
        # Let's try: Start with low damping (hot, explore), end with high damping (cool, settle).
        # Damping 0.90 (hot) to 0.999 (cold).
        damping = 0.90 + 0.09 * (phase / (num_phases - 1))
        
        # Reduce dt slightly for stability as forces might get large? 
        # Or keep dt constant.
        
        for step in range(steps_per_phase):
            if time.time() > deadline:
                break
            
            # 1. Compute Forces
            # Pairwise
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            # Avoid division by zero
            dist_sq[dist_sq == 0] = 1e-12
            dist = np.sqrt(dist_sq)
            
            # Force magnitude: k * (1/d^2) or similar. 
            # Let's use a soft repulsion: k * (1 - d)^2 if d < 1? No, we want them to push apart always to maximize min dist.
            # Standard LJ-like or simply inverse square.
            # F = k * diff / dist^3  (Vector form of k/d^2 * unit_vec)
            # To avoid explosion, clip force.
            
            inv_dist3 = 1.0 / (dist_sq * dist)
            # Mask diagonal
            inv_dist3[np.diag_indices(N)] = 0.0
            
            # Force matrix
            force_matrix = k_rep * inv_dist3[:, :, np.newaxis] * diff
            forces = np.sum(force_matrix, axis=1)
            
            # Boundary Force
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            # Push inward if r > 0.9 (soft wall)
            # Force = -k_wall * (r - r_wall) * unit_r
            # Let r_wall = 0.95 to keep them well inside initially, then tighten?
            # Actually, the constraint is r <= 1.
            # If we just maximize min dist, the boundary will naturally constrain them.
            # But to prevent escape, we need a hard wall or strong soft wall.
            # Let's use a quadratic penalty for r > 0.99
            wall_force = np.zeros_like(pts)
            mask_wall = r_pts > 0.99
            if np.any(mask_wall):
                # Strength increases quadratically with violation
                violations = r_pts[mask_wall] - 0.99
                unit_vec = pts[mask_wall] / r_pts[mask_wall, np.newaxis]
                wall_force[mask_wall] = -k_wall * violations * unit_vec
            
            forces += wall_force

            # 2. Update Velocity (Velocity Verlet half-step)
            velocities += forces * dt * 0.5
            velocities *= damping
            
            # 3. Update Position
            pts += velocities * dt
            
            # 4. Update Velocity again (second half-step)
            # Re-calculate forces for new positions? 
            # For simplicity and speed, let's do Euler-Cromer (semi-implicit) which is stable enough for this.
            # pts += v * dt
            # v = (v + F * dt) * damping
            # The above code did:
            # v += 0.5 F dt
            # v *= damp
            # p += v dt
            # This is slightly off from standard Verlet. Let's stick to a simple Symplectic Euler:
            # p += v dt
            # v = (v + F dt) * damp
            # I already updated p. Let's redo the loop logic to be cleaner Symplectic Euler.
            
            # Reset for next iteration if we used the Verlet-like update above? 
            # The code above did:
            # 1. Calc F
            # 2. v += 0.5 F dt; v *= damp
            # 3. p += v dt
            # This is a valid symplectic integrator variant. 
            # We don't strictly need the second half-step force eval if we accept the slight energy drift, 
            # especially with damping.
            
            # Check for NaNs
            if not np.all(np.isfinite(pts)):
                pts = best_pts
                velocities = np.zeros_like(pts)
                break

        # After phase, check improvement
        # Project points inside disk if they drifted out
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = r_pts > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] / r_pts[mask_out, np.newaxis] * 0.9999
        
        current_radius = compute_radius(pts)
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = pts.copy()

    # Final Polish: Use the MD result as start for a quick gradient ascent on the min distance?
    # Or just return the MD result. MD usually finds very good local minima.
    
    # One more refinement: Local search on the best points found
    if time.time() < deadline - 5:
        # Simple hill climbing on the min distance
        current_pts = best_pts.copy()
        current_r = best_radius
        step_size = 0.001
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0: continue
                        new_pts = current_pts.copy()
                        new_pts[i, 0] += dx
                        new_pts[i, 1] += dy
                        # Check boundary
                        r = np.hypot(new_pts[i, 0], new_pts[i, 1])
                        if r > 1.0:
                            continue
                        new_r = compute_radius(new_pts)
                        if new_r > current_r:
                            current_pts = new_pts
                            current_r = new_r
                            improved = True
            if not improved:
                step_size *= 0.7
                if step_size < 1e-5:
                    break
            if current_r > best_radius:
                best_radius = current_r
                best_pts = current_pts.copy()

    # Final validation
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    return best_pts
