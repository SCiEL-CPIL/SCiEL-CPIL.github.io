"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy: Continuous Simulated Annealing
    - Circles have continuous positions (x, y) and radii r.
    - Objective: Maximize sum(r) subject to non-overlap and boundary constraints.
    - Moves: Small Gaussian perturbations to x, y, r.
    - Aspect ratio: Optimized by running SA for multiple fixed aspect ratios.
    - Evaluation: Vectorized distance calculation for efficiency.
    """
    n = N
    best_global_score = -float('inf')
    best_global_pos = None
    best_global_rad = None
    best_global_w = 1.0
    best_global_h = 1.0
    
    # Perimeter is 4, so w + h = 2.
    # Try various aspect ratios w/h
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5]
    
    for aspect in aspect_ratios:
        h = 2.0 / (1.0 + aspect)
        w = aspect * h
        
        if w < 0.1 or h < 0.1:
            continue
            
        # 1. Initialization: Hexagonal packing
        # Create a hexagonal grid of points
        # Approximate radius for 21 circles in area w*h:
        # Area per circle ~ pi*r^2. Total area ~ n*pi*r^2.
        # Packing density ~ 0.9.
        # n*pi*r^2 ~ 0.9 * w * h
        # r ~ sqrt(0.9 * w * h / (n * pi))
        r_est = np.sqrt(0.9 * w * h / (n * np.pi))
        
        # Generate hex grid points
        # Spacing s = 2 * r_est
        s = 2.0 * r_est
        if s < 1e-6:
            s = 0.1
            
        points = []
        # y step is s * sqrt(3)/2
        dy = s * np.sqrt(3) / 2
        
        gy = 0
        while gy < h:
            gx = 0
            offset = 0
            if int(gy / dy) % 2 == 1:
                offset = s / 2
                
            while gx < w:
                points.append((gx + offset, gy))
                gx += s
            gy += dy
            
        # Shuffle and pick n points? Or just take first n?
        # Taking first n might cluster. Better to sample from the grid or use a random subset.
        # Let's just take the first n points that fit well, or random sample if grid is large.
        if len(points) >= n:
            # Randomly sample n points from the grid to avoid bias
            indices = np.random.choice(len(points), n, replace=False)
            pos = np.array([points[i] for i in indices])
        else:
            # If grid is too sparse, add random points
            pos = np.array(points)
            while len(pos) < n:
                pos = np.vstack([pos, np.array([np.random.uniform(0, w), np.random.uniform(0, h)])])
            pos = pos[:n]
            
        # Initial radii
        rad = np.full(n, r_est)
        
        # Ensure initial validity (clamp positions)
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
            
        # 2. Simulated Annealing
        T = 0.1
        T_min = 0.001
        cooling = 0.999
        max_iters = 20000
        rng = np.random.default_rng(42)
        
        current_pos = pos.copy()
        current_rad = rad.copy()
        
        def evaluate(pos, rad):
            # pos: (n, 2), rad: (n,)
            # Check bounds
            # x < r or x > w-r -> penalty
            # y < r or y > h-r -> penalty
            # Pairwise overlap
            # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            # dist = np.sqrt(np.sum(diff**2, axis=-1))
            # min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
            # overlap = np.maximum(0, min_dist - dist)
            # penalty = np.sum(overlap**2)
            # score = np.sum(rad) - penalty
            # Also add boundary penalty if any
            # bx = np.maximum(0, rad - pos[:, 0]) + np.maximum(0, pos[:, 0] - (w - rad))
            # by = np.maximum(0, rad - pos[:, 1]) + np.maximum(0, pos[:, 1] - (h - rad))
            # bpen = np.sum(bx**2) + np.sum(by**2)
            
            # Vectorized calculation
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Mask diagonal
            np.fill_diagonal(dist, np.inf)
            
            min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
            
            overlap = np.maximum(0, min_dist - dist)
            # Zero out diagonal
            np.fill_diagonal(overlap, 0)
            
            penalty_overlap = np.sum(overlap**2)
            
            # Boundary penalty
            bx = np.maximum(0, rad - pos[:, 0]) + np.maximum(0, pos[:, 0] - (w - rad))
            by = np.maximum(0, rad - pos[:, 1]) + np.maximum(0, pos[:, 1] - (h - rad))
            penalty_bound = np.sum(bx**2) + np.sum(by**2)
            
            total_penalty = penalty_overlap + penalty_bound
            score = np.sum(rad) - total_penalty
            
            return score

        current_score = evaluate(current_pos, current_rad)
        best_local_score = current_score
        best_local_pos = current_pos.copy()
        best_local_rad = current_rad.copy()
        
        step_size = 0.1
        
        for it in range(max_iters):
            # Generate move
            i = rng.integers(n)
            
            # Perturb position
            dx = rng.normal(0, step_size)
            dy = rng.normal(0, step_size)
            
            # Perturb radius
            dr = rng.normal(0, step_size * 0.5)
            
            new_pos = current_pos.copy()
            new_rad = current_rad.copy()
            
            new_pos[i, 0] += dx
            new_pos[i, 1] += dy
            new_rad[i] += dr
            
            # Clamp radius
            if new_rad[i] < 0.01:
                new_rad[i] = 0.01
            if new_rad[i] > min(w, h) / 2:
                new_rad[i] = min(w, h) / 2
                
            # Clamp position to stay within bounds relative to radius
            new_pos[i, 0] = np.clip(new_pos[i, 0], new_rad[i], w - new_rad[i])
            new_pos[i, 1] = np.clip(new_pos[i, 1], new_rad[i], h - new_rad[i])
            
            new_score = evaluate(new_pos, new_rad)
            
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_pos = current_pos.copy()
                    best_local_rad = current_rad.copy()
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
            # Adapt step size
            if it % 1000 == 0:
                step_size *= 0.95
                
        # Update global best
        if best_local_score > best_global_score:
            best_global_score = best_local_score
            best_global_pos = best_local_pos
            best_global_rad = best_local_rad
            best_global_w = w
            best_global_h = h
            
    # Construct output
    if best_global_pos is None:
        # Fallback
        return np.zeros((n, 3))
        
    result = np.zeros((n, 3))
    for i in range(n):
        result[i, 0] = best_global_pos[i, 0]
        result[i, 1] = best_global_pos[i, 1]
        result[i, 2] = best_global_rad[i]
        
    # Final validation and cleanup
    # Ensure no overlaps and within bounds
    # If there are small overlaps due to floating point or SA noise, scale down radii
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((result[i, 0] - result[j, 0])**2 + (result[i, 1] - result[j, 1])**2)
            if dist < result[i, 2] + result[j, 2] - 1e-6:
                valid = False
        if result[i, 0] < result[i, 2] or result[i, 0] > best_global_w - result[i, 2]:
            valid = False
        if result[i, 1] < result[i, 2] or result[i, 1] > best_global_h - result[i, 2]:
            valid = False
            
    if not valid:
        # Scale down radii to fix overlaps
        scale = 1.0
        while scale > 0.1:
            scale -= 0.01
            temp_result = result.copy()
            for i in range(n):
                temp_result[i, 2] *= scale
                temp_result[i, 0] = np.clip(temp_result[i, 0], temp_result[i, 2], best_global_w - temp_result[i, 2])
                temp_result[i, 1] = np.clip(temp_result[i, 1], temp_result[i, 2], best_global_h - temp_result[i, 2])
                
            # Check validity
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((temp_result[i, 0] - temp_result[j, 0])**2 + (temp_result[i, 1] - temp_result[j, 1])**2)
                    if dist < temp_result[i, 2] + temp_result[j, 2] - 1e-6:
                        ok = False
                if temp_result[i, 0] < temp_result[i, 2] or temp_result[i, 0] > best_global_w - temp_result[i, 2]:
                    ok = False
                if temp_result[i, 1] < temp_result[i, 2] or temp_result[i, 1] > best_global_h - temp_result[i, 2]:
                    ok = False
            if ok:
                result = temp_result
                break
                
    return result
# EVOLVE-BLOCK-END
