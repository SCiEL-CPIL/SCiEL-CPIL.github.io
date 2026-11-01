"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_dist = min(min_dist, np.min(d) if len(d) > 0 else np.inf)
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    def physics_simulate(pts, max_iters=5000, dt=0.01, cooling_rate=0.999):
        """Physics-based simulation with repulsive forces between circles and container wall."""
        pts = pts.copy()
        n = len(pts)
        
        # Estimate initial radius for force calculations
        min_dist = np.inf
        for i in range(n):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_dist = min(min_dist, np.min(d) if len(d) > 0 else np.inf)
        r_est = min_dist / 2.0
        
        velocity = np.zeros_like(pts)
        temperature = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces (only for close pairs to save time)
            cutoff = 3.0 * r_est
            for i in range(n):
                # Compute distances to all other points
                diffs = pts[i+1:] - pts[i]
                dists = np.linalg.norm(diffs, axis=1)
                
                # Only consider pairs within cutoff
                mask = dists < cutoff
                if not np.any(mask):
                    continue
                
                close_diffs = diffs[mask]
                close_dists = dists[mask]
                
                # Repulsive force: F = k / d^2 (inverse square)
                # This creates strong repulsion at close range
                k = 0.1
                force_magnitudes = k / (close_dists ** 2 + 1e-10)
                force_vectors = close_diffs / (close_dists[:, np.newaxis] + 1e-10) * force_magnitudes[:, np.newaxis]
                
                forces[i] -= np.sum(force_vectors, axis=0)
                
                # Apply equal and opposite force to other particles
                for j in range(len(force_vectors)):
                    idx = i + 1 + np.where(mask)[0][j]
                    forces[idx] += force_vectors[j]
            
            # Container wall force: push inward if too close to boundary
            dists_to_origin = np.linalg.norm(pts, axis=1)
            wall_margin = 0.1  # Start pushing when within this distance of wall
            wall_force_strength = 1.0
            for i in range(n):
                if dists_to_origin[i] > 1.0 - wall_margin:
                    # Push inward proportionally to how far past margin
                    overshoot = dists_to_origin[i] - (1.0 - wall_margin)
                    if overshoot > 0:
                        direction = pts[i] / (dists_to_origin[i] + 1e-10)
                        forces[i] -= direction * wall_force_strength * overshoot
            
            # Update velocity with damping
            damping = 0.9
            velocity = velocity * damping + forces * dt
            
            # Limit velocity to prevent instability
            max_vel = 0.05
            vel_mags = np.linalg.norm(velocity, axis=1)
            for i in range(n):
                if vel_mags[i] > max_vel:
                    velocity[i] *= max_vel / vel_mags[i]
            
            # Update positions
            pts += velocity * dt
            
            # Project back inside container if necessary
            dists_to_origin = np.linalg.norm(pts, axis=1)
            for i in range(n):
                if dists_to_origin[i] > 1.0:
                    pts[i] *= 1.0 / dists_to_origin[i]
            
            # Cool down
            temperature *= cooling_rate
        
        return pts
    
    def local_optimize(pts, max_iters=200, lr=0.001):
        """Local optimization using gradient ascent on a smooth objective."""
        pts = pts.copy()
        n = len(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current score
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                # Push back inside
                for i in range(n):
                    if dists[i] > 1.0:
                        pts[i] *= 1.0 / dists[i]
                    else:
                        dists = np.linalg.norm(pts, axis=1)
                continue
            
            # Find the minimum pairwise distance and which pair
            min_dist = np.inf
            min_i, min_j = 0, 1
            for i in range(n):
                d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
                if len(d) > 0:
                    idx = np.argmin(d)
                    if d[idx] < min_dist:
                        min_dist = d[idx]
                        min_i, min_j = i, i + 1 + idx
            
            r_pair = min_dist / 2.0
            r_wall = 1.0 - np.max(dists)
            
            # Determine which constraint is binding
            if r_pair < r_wall:
                # Pairwise constraint is binding: push the pair apart
                diff = pts[min_j] - pts[min_i]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    direction = diff / dist
                    pts[min_i] -= direction * lr
                    pts[min_j] += direction * lr
            else:
                # Wall constraint is binding: push the outermost point inward
                outer_idx = np.argmax(dists)
                direction = pts[outer_idx] / (dists[outer_idx] + 1e-10)
                pts[outer_idx] -= direction * lr
            
            # Project back inside container
            dists = np.linalg.norm(pts, axis=1)
            for i in range(n):
                if dists[i] > 1.0:
                    pts[i] *= 1.0 / dists[i]
        
        return pts
    
    # Strategy 1: Generate hexagonal lattice initialization
    def generate_hex_lattice(n_points, radius=1.0, perturbation=0.0):
        """Generate a hexagonal lattice of points inside a circle, with optional perturbation."""
        # Estimate spacing for n_points in a circle of radius 1
        # Area of circle = pi, area per hex = (sqrt(3)/2) * s^2 where s is spacing
        # n_points * (sqrt(3)/2) * s^2 ≈ pi
        s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0) / 2.0))
        
        # Generate hexagonal lattice points
        k_max = int(1.0 / s) + 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - s * 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have too many, select the ones closest to center
        if len(points) > n_points:
            dists = np.linalg.norm(points, axis=1)
            points = points[np.argsort(dists, kind='stable')[:n_points]]
        
        # If we have too few, add more with smaller spacing
        while len(points) < n_points:
            s *= 0.95
            points_list = []
            k_max = int(1.0 / s) + 2
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 1.0 - s * 0.5:
                        points_list.append([x, y])
            points = np.array(points_list)
            if len(points) > n_points:
                dists = np.linalg.norm(points, axis=1)
                points = points[np.argsort(dists, kind='stable')[:n_points]]
        
        # Add perturbation to break symmetry
        if perturbation > 0 and len(points) == n_points:
            rng = np.random.default_rng(42)
            points += rng.normal(0, perturbation, size=points.shape)
            # Project back inside container
            dists = np.linalg.norm(points, axis=1)
            for i in range(len(points)):
                if dists[i] > 1.0:
                    points[i] *= 1.0 / dists[i]
        
        return points
    
    # Generate initial hexagonal lattice with perturbation
    hex_points = generate_hex_lattice(N, perturbation=0.01)
    best_score = compute_score(hex_points)
    best_pts = hex_points.copy()
    
    # Strategy 2: Run physics simulation
    if time.time() < deadline - 30:
        sim_pts = physics_simulate(hex_points, max_iters=3000, dt=0.005, cooling_rate=0.9995)
        sim_score = compute_score(sim_pts)
        if sim_score > best_score:
            best_score = sim_score
            best_pts = sim_pts
    
    # Strategy 3: Local optimization on the best configuration
    if time.time() < deadline - 10 and best_pts is not None:
        opt_pts = local_optimize(best_pts, max_iters=100, lr=0.0005)
        opt_score = compute_score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Strategy 4: Try multiple perturbations and pick the best
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            # Generate new perturbed lattice
            perturbed = generate_hex_lattice(N, perturbation=0.015)
            # Add additional random perturbation
            perturbed += rng.normal(0, 0.01, size=perturbed.shape)
            dists = np.linalg.norm(perturbed, axis=1)
            for i in range(N):
                if dists[i] > 1.0:
                    perturbed[i] *= 1.0 / dists[i]
            
            # Run physics simulation on this perturbation
            sim_pts = physics_simulate(perturbed, max_iters=2000, dt=0.005, cooling_rate=0.9995)
            sim_score = compute_score(sim_pts)
            
            # Local optimize
            if time.time() < deadline - 10:
                opt_pts = local_optimize(sim_pts, max_iters=50, lr=0.0005)
                opt_score = compute_score(opt_pts)
                if opt_score > sim_score:
                    sim_score = opt_score
                    sim_pts = opt_pts
            
            if sim_score > best_score:
                best_score = sim_score
                best_pts = sim_pts
    
    # Final validation
    if best_pts is None or compute_score(best_pts) <= 0:
        # Return a valid fallback
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing
    
    return best_pts
# EVOLVE-BLOCK-END
