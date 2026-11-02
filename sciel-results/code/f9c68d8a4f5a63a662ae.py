"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Physics-based relaxation (Molecular Dynamics).
    1. Initialize circles in a grid.
    2. Simulate repulsive forces to resolve overlaps.
    3. Gradually increase the target radius until the system stabilizes at a maximum packing.
    4. Try multiple aspect ratios to find the global optimum.
    """
    best_result = None
    best_score = -np.inf
    
    # Try several aspect ratios. 
    # For N=21, a square or slightly rectangular box is often optimal.
    # w + h = 2.
    aspects = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    for aspect in aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Initialize positions in a grid
        # We want to fit 21 circles. A 5x5 grid has 25 slots. 
        # We can use a 5x5 grid and remove 4, or just place 21 in a 5x5 grid pattern.
        # Let's use a 5x5 grid and take the first 21 positions.
        cols = 5
        rows = 5
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            # Center of each cell
            positions[i, 0] = (col + 0.5) * w / cols
            positions[i, 1] = (row + 0.5) * h / rows
            
        velocities = np.zeros((N, 2))
        
        # Start with a small radius and grow it
        # Estimate max possible radius: Area / N roughly.
        # Area = w*h. r_max ~ sqrt(w*h / N)
        r_init = 0.01
        r_step = 0.005
        r_current = r_init
        
        # Physics parameters
        damping = 0.9
        repulsion_strength = 50.0
        max_iter = 10000
        stability_counter = 0
        stable_threshold = 50
        
        # We will run the simulation for a fixed number of steps or until stable.
        # To maximize sum, we increase r_current if the system is stable.
        
        for step in range(max_iter):
            forces = np.zeros((N, 2))
            
            # 1. Repulsion between circles
            # Vectorized distance calculation
            # diff[i, j] = pos[i] - pos[j]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Overlap condition: dist < 2 * r_current
            # Force is along the vector, magnitude proportional to overlap
            # We only care about i < j to avoid double counting and self-interaction
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    min_dist = 2.0 * r_current
                    if d < min_dist:
                        # Overlap amount
                        overlap = min_dist - d
                        # Force direction: from j to i (repel i away from j)
                        # diff[i, j] is pos[i] - pos[j]
                        # Normalized direction
                        if d > 1e-6:
                            dir_vec = diff[i, j] / d
                        else:
                            # Random direction if exactly same position
                            dir_vec = np.random.randn(2)
                            dir_vec /= np.linalg.norm(dir_vec)
                        
                        # Force magnitude
                        f_mag = repulsion_strength * overlap
                        forces[i] += f_mag * dir_vec
                        forces[j] -= f_mag * dir_vec
            
            # 2. Boundary constraints
            # If a circle goes out of bounds, push it back
            for i in range(N):
                x, y = positions[i]
                r = r_current
                
                # Left wall
                if x < r:
                    forces[i, 0] += repulsion_strength * (r - x)
                # Right wall
                if x > w - r:
                    forces[i, 0] -= repulsion_strength * (x - (w - r))
                # Bottom wall
                if y < r:
                    forces[i, 1] += repulsion_strength * (y - r)
                # Top wall
                if y > h - r:
                    forces[i, 1] -= repulsion_strength * (y - (h - r))
            
            # Update velocities and positions (Semi-implicit Euler / Verlet-like)
            velocities += forces
            velocities *= damping
            positions += velocities
            
            # Check for stability: if no overlaps and no boundary violations
            # We check this cheaply by looking at max overlap
            min_dist_matrix = 2.0 * r_current
            # Re-calculate distances for stability check? Or reuse?
            # Reusing dist from above is fine as it's updated.
            # But we need to check if ANY pair is overlapping.
            # dist is symmetric. 
            # We can check np.min(dist) > min_dist_matrix
            # However, dist[i,i] is 0. So we need to ignore diagonal.
            
            # Efficient check:
            # dist[~np.eye(N, dtype=bool)] gives all off-diagonal distances.
            min_off_diag_dist = np.min(dist[~np.eye(N, dtype=bool)])
            
            # Check boundaries
            in_bounds = True
            for i in range(N):
                x, y = positions[i]
                if x < r_current - 1e-6 or x > w - r_current + 1e-6 or \
                   y < r_current - 1e-6 or y > h - r_current + 1e-6:
                    in_bounds = False
                    break
            
            if min_off_diag_dist > min_dist_matrix and in_bounds:
                stability_counter += 1
            else:
                stability_counter = 0
                
            # If stable for a while, try to increase radius
            if stability_counter > stable_threshold:
                r_current += r_step
                stability_counter = 0
                # Small perturbation to escape local minima
                positions += np.random.randn(N, 2) * 0.001
                velocities = np.zeros((N, 2))
                
            # If radius grows too large such that it definitely doesn't fit (heuristic), stop
            # Max radius is limited by area: N * pi * r^2 <= w * h
            # r_max_area = sqrt(w * h / (N * pi))
            if r_current > np.sqrt(w * h / (N * np.pi)) * 1.2:
                break

        # Final verification and scoring
        # The current r_current is the radius we ended up with.
        # We must ensure the final configuration is valid for this r_current.
        # If the last step increased r_current, the system might be unstable.
        # We should do a few final relaxation steps with the final r_current.
        
        # Final relaxation
        for _ in range(200):
            forces = np.zeros((N, 2))
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = dist[i, j]
                    min_dist = 2.0 * r_current
                    if d < min_dist:
                        overlap = min_dist - d
                        if d > 1e-6:
                            dir_vec = diff[i, j] / d
                        else:
                            dir_vec = np.random.randn(2)
                            dir_vec /= np.linalg.norm(dir_vec)
                        f_mag = repulsion_strength * overlap
                        forces[i] += f_mag * dir_vec
                        forces[j] -= f_mag * dir_vec
            
            for i in range(N):
                x, y = positions[i]
                r = r_current
                if x < r: forces[i, 0] += repulsion_strength * (r - x)
                if x > w - r: forces[i, 0] -= repulsion_strength * (x - (w - r))
                if y < r: forces[i, 1] += repulsion_strength * (y - r)
                if y > h - r: forces[i, 1] -= repulsion_strength * (y - (h - r))
            
            velocities += forces
            velocities *= damping
            positions += velocities

        # Check validity
        valid = True
        # Check overlaps
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = 2.0 * r_current
        # Check all pairs
        for i in range(N):
            for j in range(i+1, N):
                if dist[i, j] < min_dist - 1e-6:
                    valid = False
                    break
            if not valid: break
        
        # Check bounds
        if valid:
            for i in range(N):
                x, y = positions[i]
                r = r_current
                if x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
                    valid = False
                    break
        
        if valid:
            score = N * r_current
            if score > best_score:
                best_score = score
                best_result = np.column_stack((positions[:, 0], positions[:, 1], np.full(N, r_current)))
    
    if best_result is None:
        # Fallback: Grid packing with small radius
        w, h = 1.0, 1.0
        cols = 5
        rows = 5
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            positions[i, 0] = (col + 0.5) * w / cols
            positions[i, 1] = (row + 0.5) * h / rows
        r = 0.05
        best_result = np.column_stack((positions[:, 0], positions[:, 1], np.full(N, r)))
        
    return best_result
# EVOLVE-BLOCK-END
