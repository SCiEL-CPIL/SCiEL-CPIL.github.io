"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n = N
    grid_res = 60  # Higher resolution for better precision
    rng = random.Random(42)
    
    def get_coords(gx, gy, w, h):
        cell_w = w / grid_res
        cell_h = h / grid_res
        x = (gx + 0.5) * cell_w
        y = (gy + 0.5) * cell_h
        return x, y

    def compute_score(grid_positions, w, h):
        """
        Compute the sum of radii for a given set of grid positions.
        Uses the conservative heuristic: r_i = 0.5 * min(2*boundary_limit_i, min_neighbor_dist_i)
        This guarantees feasibility.
        """
        positions = np.array([get_coords(gx, gy, w, h) for gx, gy in grid_positions])
        n = len(positions)
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((positions[i,0] - positions[j,0])**2 + (positions[i,1] - positions[j,1])**2)
                dists[i,j] = d
                dists[j,i] = d
        
        # Calculate boundary limits
        boundary_limits = np.zeros(n)
        for i in range(n):
            x, y = positions[i]
            boundary_limits[i] = min(x, w-x, y, h-y)
        
        # Calculate min neighbor distance
        min_neighbor_dists = np.full(n, float('inf'))
        for i in range(n):
            for j in range(n):
                if i != j and dists[i,j] < min_neighbor_dists[i]:
                    min_neighbor_dists[i] = dists[i,j]
        
        # Calculate radii
        radii = np.zeros(n)
        for i in range(n):
            if min_neighbor_dists[i] == float('inf'):
                radii[i] = boundary_limits[i]
            else:
                radii[i] = 0.5 * min(2 * boundary_limits[i], min_neighbor_dists[i])
        
        return np.sum(radii), radii

    best_score_global = -float('inf')
    best_solution_global = None
    
    # Try a range of aspect ratios
    # t is deviation from square. w = 1+t, h = 1-t. Perimeter = 2(w+h) = 4.
    aspect_ratios = np.linspace(-0.3, 0.3, 13)
    
    for t in aspect_ratios:
        w = 1.0 + t
        h = 1.0 - t
        if w <= 0.1 or h <= 0.1: continue
        
        # Initialize with a hexagonal-like pattern
        circles = []
        occupied = set()
        # Start from center and spiral out? Or just scan?
        # Scanning is simple. Let's do a dense scan.
        for gy in range(grid_res):
            for gx in range(grid_res):
                if len(circles) >= n: break
                if (gx, gy) not in occupied:
                    circles.append((gx, gy))
                    occupied.add((gx, gy))
            if len(circles) >= n: break
        
        if len(circles) < n: continue
        
        current_score, _ = compute_score(circles, w, h)
        best_local_score = current_score
        best_local_circles = circles[:]
        
        # Simulated Annealing
        T = 0.5
        T_min = 0.001
        cooling = 0.9995
        max_iters = 15000
        
        for it in range(max_iters):
            move_type = rng.random()
            new_circles = circles[:]
            
            if move_type < 0.4:
                # Move one circle to a nearby cell
                i = rng.randint(0, n-1)
                gx, gy = circles[i]
                dx = rng.randint(-2, 2)
                dy = rng.randint(-2, 2)
                new_gx, new_gy = gx + dx, gy + dy
                if 0 <= new_gx < grid_res and 0 <= new_gy < grid_res:
                    if (new_gx, new_gy) not in occupied or (new_gx, new_gy) == (gx, gy):
                        new_circles[i] = (new_gx, new_gy)
                    else:
                        continue
            elif move_type < 0.7:
                # Swap two circles
                i, j = rng.sample(range(n), 2)
                new_circles[i], new_circles[j] = new_circles[j], new_circles[i]
            else:
                # Move one circle to a random cell
                i = rng.randint(0, n-1)
                new_gx, new_gy = rng.randint(0, grid_res-1), rng.randint(0, grid_res-1)
                if (new_gx, new_gy) not in occupied or (new_gx, new_gy) == circles[i]:
                    new_circles[i] = (new_gx, new_gy)
                else:
                    continue
            
            # Check for duplicates in new_circles
            if len(set(new_circles)) < n:
                continue
            
            new_score, _ = compute_score(new_circles, w, h)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                # Update state
                # Rebuild occupied set
                occupied = set(circles)
                circles = new_circles
                current_score = new_score
                occupied = set(circles)
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_circles = circles[:]
            
            T = max(T * cooling, T_min)
        
        if best_local_score > best_score_global:
            best_score_global = best_local_score
            best_solution_global = (best_local_circles, w, h)
    
    if best_solution_global is None:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n: break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    best_circles, w, h = best_solution_global
    _, radii = compute_score(best_circles, w, h)
    
    result = []
    for i in range(n):
        gx, gy = best_circles[i]
        x, y = get_coords(gx, gy, w, h)
        r = radii[i]
        # Clamp to ensure validity
        r = max(0.001, min(r, min(w, h) / 2 - 0.001))
        x = max(r, min(x, w - r))
        y = max(r, min(y, h - r))
        result.append([x, y, r])
    
    result = np.array(result)
    
    # Final validation and repair
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2)
            if dist < result[i,2] + result[j,2] - 1e-7:
                valid = False
        if result[i,0] < result[i,2] or result[i,0] > w - result[i,2]:
            valid = False
        if result[i,1] < result[i,2] or result[i,1] > h - result[i,2]:
            valid = False
            
    if not valid:
        scale = 0.9
        for i in range(n):
            result[i,2] *= scale
            result[i,0] = max(result[i,2], min(result[i,0], w - result[i,2]))
            result[i,1] = max(result[i,2], min(result[i,1], h - result[i,2]))
            
    return result
# EVOLVE-BLOCK-END
