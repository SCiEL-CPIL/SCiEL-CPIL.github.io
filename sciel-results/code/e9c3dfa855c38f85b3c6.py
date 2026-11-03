import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 0.98:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) >= N:
            # Select N points using farthest-point sampling
            selected = [np.random.choice(len(pts))]
            while len(selected) < N:
                dists = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in selected], axis=0)
                next_idx = np.argmax(dists)
                selected.append(next_idx)
            pts = pts[selected]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def compute_score(pts):
        """Compute the packing radius score."""
        tree = cKDTree(pts)
        k = min(20, N - 1)
        dists_k, idx_k = tree.query(pts, k=k + 1)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(1, k + 1):
                d = dists_k[i, j]
                if d > 1e-10:
                    min_pair_dist = min(min_pair_dist, d)
        
        min_bound = np.min(bound_dists)
        return min(min_pair_dist / 2.0, min_bound)

    # Initial setup
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    pts = hex_lattice_init(base_spacing * 0.95)
    best_pts = pts.copy()
    best_score = compute_score(best_pts)
    
    # Velocity and momentum
    v = np.zeros_like(pts)
    
    # Simulation parameters
    dt = 0.005
    damping = 0.98
    target_r = best_score * 0.95
    noise_temp = 0.01
    min_noise_temp = 0.0001
    cooling_rate = 0.999
    
    # Main optimization loop
    while time.time() - start_time < time_limit:
        # Build KDTree for neighbor search
        tree = cKDTree(pts)
        
        # Forces
        forces = np.zeros_like(pts)
        
        # 1. Repulsive forces between overlapping circles
        # We only need to check neighbors within a certain distance
        # Let's query all points within 2*target_r + epsilon
        search_radius = 2 * target_r + 0.01
        neighbors = tree.query_ball_point(pts, r=search_radius)
        
        for i in range(N):
            for j in neighbors[i]:
                if i < j:
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * target_r and dist > 1e-10:
                        # Repulsive force: push apart
                        # Strength proportional to overlap
                        overlap = (2 * target_r - dist)
                        force_mag = overlap * 10.0 # Stiffness
                        f = (d / dist) * force_mag
                        forces[i] += f
                        forces[j] -= f
        
        # 2. Attractive forces to boundary
        # Push points towards the boundary if they are not close enough
        # This helps fill the container
        rad = np.linalg.norm(pts, axis=1)
        # If a point is significantly inside, push it out
        # We want the minimum distance to boundary to be large, so we want points to be at radius ~1
        # But we don't want them to cross.
        # Force towards boundary: (1 - rad) * strength * (pts / rad)
        # Actually, we want to maximize min(1-rad). So we want to minimize the "slack".
        # Let's apply a weak force pushing outwards for all points.
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        # Vector pointing outwards
        outward = pts / rad_safe[:, np.newaxis]
        # Strength: stronger if further from boundary? 
        # Or just constant weak force to expand the cluster?
        # Let's use a force that pushes points to be at radius 1.
        # F = (1 - rad) * k_bound * outward
        k_bound = 0.1
        forces += (1.0 - rad)[:, np.newaxis] * k_bound * outward
        
        # 3. Centering force (weak) to prevent drift
        forces -= pts * 0.01
        
        # 4. Random noise (Annealing)
        forces += np.random.normal(0, noise_temp, size=pts.shape)
        
        # Update velocities and positions
        v = v * damping + forces * dt
        pts = pts + v * dt
        
        # Constrain inside unit circle
        rad = np.linalg.norm(pts, axis=1)
        mask_out = rad > 1.0
        if np.any(mask_out):
            pts[mask_out] /= rad[mask_out, np.newaxis]
            v[mask_out] *= 0.5 # Dampen velocity on collision
        
        # Calculate current score
        current_score = compute_score(pts)
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            # If we improved, we can try to increase target_r slightly
            # This is a heuristic to tighten the packing
            target_r = best_score * 0.98
        else:
            # If we didn't improve, maybe we are stuck.
            # Cool down the noise
            noise_temp = max(min_noise_temp, noise_temp * cooling_rate)
            
            # If noise is very low and we haven't improved in a while, restart with higher noise?
            # For simplicity, let's just let it run. The repulsion forces will eventually find a way if target_r is high enough.
            # However, if target_r is too high, forces will be huge and chaotic.
            # Let's adjust target_r based on best_score but keep it slightly below to allow movement.
            # Actually, keeping target_r fixed relative to best_score is a good strategy.
            # If stuck, we might need to "heat up" again.
            if noise_temp < min_noise_temp:
                # Reheat slightly to escape local minima
                noise_temp = min_noise_temp * 5
                target_r = best_score * 0.95 # Reset target to allow some slack
        
        # Adaptive dt?
        # If forces are very large, reduce dt to avoid explosion
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 10.0:
            dt = 0.001
        else:
            dt = 0.005
            
        # Periodic check to ensure we are not diverging
        if not np.all(np.isfinite(pts)):
            pts = best_pts.copy()
            v = np.zeros_like(pts)
            noise_temp = 0.01
            target_r = best_score * 0.95

    # Final verification
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
