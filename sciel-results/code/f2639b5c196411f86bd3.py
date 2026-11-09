# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

N = 26

def _hex_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    # For N=26, a 5x5 grid is 25. A 5x6 grid is 30.
    # Let's create a 5x6 grid and randomly drop 4 points to get 26.
    # Or better, a 6x5 grid.
    # Hex packing is denser.
    # Let's try to construct a specific hex-like pattern.
    # Rows of 5, 6, 5, 6, 4? 5+6+5+6+4 = 26.
    # This is a good heuristic.
    
    rows = [5, 6, 5, 6, 4]
    positions = []
    total_rows = len(rows)
    # Vertical spacing
    # In hex packing, vertical distance is sqrt(3)/2 * horizontal_spacing
    # Let's just use a rectangular grid with offsets for now, it's simpler and robust.
    # Actually, let's just use a jittered 5x5 + 1 center, or 6x5 - 4.
    
    # Let's use a 6x5 grid (30 points) and randomly remove 4.
    # This ensures good initial spacing.
    cols = 6
    rows_count = 5
    pts = []
    for r in range(rows_count):
        for c in range(cols):
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows_count
            # Add hex offset
            if r % 2 == 1:
                x += 0.5 / cols
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
            
    # We have 30 points. Randomly select 26.
    indices = random.sample(range(len(pts)), n)
    return np.array([pts[i] for i in indices])

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 with disjoint interiors,
    maximizing the sum of their radii.
    
    Uses a projection-based force-directed relaxation with simulated annealing jitter.
    """
    
    best_solution = None
    best_sum = -1.0
    
    n_restarts = 15
    max_iterations = 3000
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize positions
        if restart < 3:
            positions = _hex_init(N)
        else:
            # Jittered grid
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        y = (j + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        positions.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
            if len(positions) < N:
                positions.append([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)])
            positions = np.array(positions)
        
        positions = np.array(positions, dtype=np.float64)
        radii = np.full(N, 0.05)
        
        # Simulated Annealing parameters
        temp = 0.05
        cool_rate = 0.995
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Calculate distances and overlaps
            # Vectorized pairwise distance
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist = np.sqrt(dist_sq)
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            
            # Overlap depth
            min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlap = min_dist - dist
            overlap[~mask] = 0 # Zero out diagonal and non-overlaps? No, we need max(overlap, 0)
            overlap = np.maximum(overlap, 0)
            
            # 2. Move centers to resolve overlaps
            # Force is proportional to overlap, direction is unit vector
            # Avoid division by zero
            dist_safe = np.where(dist > 1e-8, dist, 1e-8)
            unit_vec = diff / dist_safe[:, :, np.newaxis]
            
            # Force on i from j
            # F_i = sum_j (overlap_ij * unit_vec_ij)
            forces = np.sum(overlap[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Boundary constraints
            # If x < r, push right. If x > 1-r, push left.
            # We can treat boundary as a "virtual circle" outside or just clamp and add force.
            # Let's add boundary forces to the force vector.
            k_boundary = 10.0
            # Left
            viol_left = np.maximum(radii - positions[:, 0], 0)
            forces[:, 0] += k_boundary * viol_left
            # Right
            viol_right = np.maximum(positions[:, 0] - (1 - radii), 0)
            forces[:, 0] -= k_boundary * viol_right
            # Bottom
            viol_bottom = np.maximum(radii - positions[:, 1], 0)
            forces[:, 1] += k_boundary * viol_bottom
            # Top
            viol_top = np.maximum(positions[:, 1] - (1 - radii), 0)
            forces[:, 1] -= k_boundary * viol_top
            
            # Update positions
            # Use a step size that depends on the magnitude of forces?
            # Or fixed step?
            # Let's use a step size that ensures we don't overshoot too much.
            step_size = 0.5
            positions += forces * step_size
            
            # Jitter (Simulated Annealing)
            positions += np.random.normal(0, temp, positions.shape)
            
            # Clamp to boundaries (hard constraint)
            positions[:, 0] = np.clip(positions[:, 0], 0.0, 1.0)
            positions[:, 1] = np.clip(positions[:, 1], 0.0, 1.0)
            
            # 3. Grow radii
            # Calculate max possible radius for each circle
            # 1. Distance to boundary
            max_r_boundary = np.minimum(
                np.minimum(positions[:, 0], 1 - positions[:, 0]),
                np.minimum(positions[:, 1], 1 - positions[:, 1])
            )
            
            # 2. Distance to other circles
            # avail_r[i] = min_j (dist[i,j] - radii[j]) / 2
            # We use the dist calculated earlier.
            # Note: dist is updated every iteration.
            
            # (dist - radii[j]) / 2
            # For i=j, dist=0. We want min over j != i.
            # Create matrix of available radius contribution
            # term[i,j] = (dist[i,j] - radii[j]) / 2
            term = (dist - radii[np.newaxis, :]) / 2.0
            # Set diagonal to inf
            np.fill_diagonal(term, np.inf)
            max_r_inter = np.min(term, axis=1)
            
            max_r = np.minimum(max_r_boundary, max_r_inter)
            
            # Grow radii towards max_r
            # If we grow too fast, we create overlaps which are resolved next step.
            # If we grow too slow, we are inefficient.
            # Let's grow by a fraction of the gap.
            gap = max_r - radii
            # Only grow if gap is positive
            growth = np.maximum(gap * 0.5, 0)
            # Cap growth to avoid huge jumps?
            # growth = np.minimum(growth, 0.005) # Maybe not needed if step_size is small
            radii += growth
            
            # Ensure radii are non-negative
            radii = np.maximum(radii, 0.0)
            
            # Cool down
            temp *= cool_rate
            
            # Convergence check
            if iteration > 100:
                # If forces are very small and radii haven't changed much
                # We can break early?
                # For now, let it run or break if temp is very low
                if temp < 0.001:
                    break
        
        # Final Validation and Cleanup
        # We need to ensure strict non-overlap.
        # The projection method might leave tiny overlaps due to numerical errors or aggressive growth.
        # Perform a few iterations of "shrink to fit" if overlaps exist.
        
        valid = True
        # Check boundaries
        for i in range(N):
            if positions[i, 0] < radii[i] - 1e-9 or positions[i, 0] > 1 - radii[i] + 1e-9:
                # Adjust radius to fit boundary
                radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
            if positions[i, 1] < radii[i] - 1e-9 or positions[i, 1] > 1 - radii[i] + 1e-9:
                radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        
        # Check pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist:
                    # Overlap exists
                    # Shrink both radii to resolve
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    if overlap > 0:
                        # Shrink each by half the overlap
                        shrink_i = overlap / 2
                        shrink_j = overlap / 2
                        radii[i] -= shrink_i
                        radii[j] -= shrink_j
                        # Ensure non-negative
                        radii[i] = max(radii[i], 0.0)
                        radii[j] = max(radii[j], 0.0)
                        valid = False # Mark as modified
        
        # Re-check after shrinking
        if not valid:
             # Simple re-check
             for i in range(N):
                 for j in range(i + 1, N):
                     dx = positions[i, 0] - positions[j, 0]
                     dy = positions[i, 1] - positions[j, 1]
                     dist_sq = dx * dx + dy * dy
                     min_dist = radii[i] + radii[j]
                     if dist_sq < min_dist * min_dist - 1e-9:
                         valid = False
                         break
                 if not valid: break
             
             # Also check boundaries again
             for i in range(N):
                 if radii[i] < 0:
                     radii[i] = 0
                 if positions[i, 0] < radii[i] - 1e-9:
                     radii[i] = min(radii[i], positions[i, 0])
                 if positions[i, 0] > 1 - radii[i] + 1e-9:
                     radii[i] = min(radii[i], 1 - positions[i, 0])
                 if positions[i, 1] < radii[i] - 1e-9:
                     radii[i] = min(radii[i], positions[i, 1])
                 if positions[i, 1] > 1 - radii[i] + 1e-9:
                     radii[i] = min(radii[i], 1 - positions[i, 1])
        
        if valid:
            total = np.sum(radii)
            if total > best_sum:
                best_sum = total
                best_solution = np.column_stack([positions[:, 0], positions[:, 1], radii])
    
    # Fallback
    if best_solution is None:
        # Simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        if len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
