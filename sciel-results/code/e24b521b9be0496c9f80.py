"""Packing 200 equal circles in a unit square using rigid-body mechanics relaxation."""
import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    # --- Hexagonal lattice initialization ---
    # For 200 circles, a hexagonal packing is denser than square.
    # We'll create a hex lattice and then relax it.
    
    def hex_init(n_circles):
        """Generate hexagonal lattice points, trimmed to n_circles."""
        points = []
        # Hex lattice: rows offset by half spacing
        # spacing s = 1.0 (will be scaled later)
        s = 1.0
        dx = s
        dy = s * np.sqrt(3) / 2
        rows = int(np.ceil(np.sqrt(n_circles / 0.866)))  # hex is denser
        for row in range(rows):
            for col in range(rows + 1):
                x = col * dx + (0.5 * dx if row % 2 else 0.0)
                y = row * dy
                points.append([x, y])
                if len(points) >= n_circles:
                    break
            if len(points) >= n_circles:
                break
        pts = np.array(points[:n_circles])
        # Center and scale to fit in [-0.5, 0.5]^2
        pts -= pts.mean(axis=0)
        max_extent = np.max(np.abs(pts))
        if max_extent > 0:
            pts *= 0.45 / max_extent  # leave some margin
        return pts

    # Multiple random initializations for diversity
    def random_init(n_circles, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        pts = rng.uniform(-0.45, 0.45, size=(n_circles, 2))
        return pts

    def compute_radius(pts):
        """Compute the maximum valid radius for given centers."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Exclude self
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        wall_dists = np.min(np.stack([
            0.5 - pts[:, 0],
            pts[:, 0] + 0.5,
            0.5 - pts[:, 1],
            pts[:, 1] + 0.5
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def relax(pts, max_steps=50000, dt=0.001, damping=0.98):
        """Velocity-Verlet style relaxation with repulsive forces."""
        pts = pts.copy()
        velocities = np.zeros_like(pts)
        
        # Repulsive potential: soft repulsion for all pairs
        # Force ~ 1/r^2 for r < threshold, 0 otherwise
        cutoff = 0.15  # interaction range
        
        best_radius = compute_radius(pts)
        best_pts = pts.copy()
        
        for step in range(max_steps):
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Repulsive force: F = (1/r^2 - 1/r0^2) * r_hat for r < r0
            r0 = 0.08  # equilibrium distance
            mask = dists < cutoff
            mask &= ~np.eye(N, dtype=bool)
            
            for i in range(N):
                for j in range(i+1, N):
                    if mask[i, j]:
                        d = dists[i, j]
                        if d < 1e-10:
                            # Jitter
                            forces[i] += np.random.uniform(-1, 1, 2) * 0.01
                            forces[j] -= np.random.uniform(-1, 1, 2) * 0.01
                        else:
                            # Repulsive force magnitude
                            f_mag = (1.0 / (d * d) - 1.0 / (r0 * r0)) if d < r0 else 0.0
                            if d > r0:
                                f_mag = -0.5 * (d - r0)  # mild attraction to maintain packing
                            f_vec = (diff[i, j] / d) * f_mag
                            forces[i] += f_vec
                            forces[j] -= f_vec
            
            # Wall forces
            for i in range(N):
                for axis in range(2):
                    if pts[i, axis] > 0.5 - 0.02:
                        forces[i, axis] -= 10.0 * (pts[i, axis] - (0.5 - 0.02))
                    elif pts[i, axis] < -0.5 + 0.02:
                        forces[i, axis] += 10.0 * ((-0.5 + 0.02) - pts[i, axis])
            
            # Velocity Verlet update
            velocities = velocities * damping + forces * dt
            pts = pts + velocities * dt
            
            # Clamp to square
            pts = np.clip(pts, -0.499, 0.499)
            
            # Track best
            current_radius = compute_radius(pts)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            
            # Early stopping if converged
            if step > 1000 and np.max(np.abs(velocities)) < 1e-6:
                break
        
        return best_pts, best_radius

    def optimize_with_adaptive_r(pts, max_iter=2000):
        """Two-phase optimization: coarse then fine radius adjustment."""
        best_pts = pts.copy()
        best_r = compute_radius(pts)
        
        # Phase 1: Coarse - increase radius and relax
        r = best_r * 1.01
        for phase in range(2):
            step_size = 0.005 if phase == 0 else 0.001
            for it in range(max_iter // 2):
                if time.time() > deadline - 5.0:
                    break
                
                # Try to increase radius
                new_r = best_r + step_size
                # Scale points to try new radius
                scale = new_r / best_r
                scaled_pts = best_pts * scale
                scaled_pts = np.clip(scaled_pts, -0.499, 0.499)
                
                # Relax
                relaxed_pts, relaxed_r = relax(scaled_pts, max_steps=2000, dt=0.0005, damping=0.97)
                
                if relaxed_r > best_r:
                    best_r = relaxed_r
                    best_pts = relaxed_pts
                else:
                    # Try smaller step
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
        
        return best_pts, best_r

    # Main optimization loop
    best_pts = None
    best_radius = 0.0
    
    # Try multiple initializations
    initializations = []
    
    # Hex initialization
    initializations.append(hex_init(N))
    
    # Random initializations
    for seed in range(5):
        initializations.append(random_init(N, seed=seed))
    
    for init_pts in initializations:
        if time.time() > deadline - 10.0:
            break
        
        # Relax from this initialization
        relaxed_pts, relaxed_r = relax(init_pts, max_steps=30000, dt=0.001, damping=0.98)
        
        if relaxed_r > best_radius:
            best_radius = relaxed_r
            best_pts = relaxed_pts
        
        # Further optimization
        if time.time() > deadline - 10.0:
            break
        
        opt_pts, opt_r = optimize_with_adaptive_r(relaxed_pts, max_iter=1000)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_pts = opt_pts
    
    # Final check
    if best_pts is None:
        best_pts = hex_init(N)
        best_radius = compute_radius(best_pts)
    
    # Ensure validity: clip to square
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
