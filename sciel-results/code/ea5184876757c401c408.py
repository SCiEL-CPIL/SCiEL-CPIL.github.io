"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def compute_radius_and_bottleneck(centers):
        """Compute radius and identify the bottleneck (closest pair or boundary violation)."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        min_wall = wall_dists.flat[min_wall_idx]
        
        # Compute pairwise distances using vectorized approach
        n = len(centers)
        # Use scipy's cdist for efficiency
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_idx = np.argmin(dist_matrix)
        min_pair = dist_matrix.flat[min_pair_idx]
        
        # Determine which is the bottleneck
        if min_wall < min_pair / 2.0:
            # Wall is the bottleneck
            bottleneck_type = 'wall'
            bottleneck_idx = min_wall_idx // 2  # index of the point
            bottleneck_value = min_wall
        else:
            # Pair is the bottleneck
            i, j = np.unravel_index(min_pair_idx, (n, n))
            bottleneck_type = 'pair'
            bottleneck_idx = (i, j)
            bottleneck_value = min_pair / 2.0
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r), bottleneck_type, bottleneck_idx, bottleneck_value
    
    def hexagonal_initialization(n_rows, n_cols, d_scale=1.0):
        """Create hexagonal packing layout."""
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts
    
    def square_grid_init(m):
        """Create square grid initialization."""
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    def bottleneck_elimination(centers, max_iterations=200):
        """Iteratively move the bottleneck point to increase the minimum radius."""
        centers = centers.copy()
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            r, btype, bidx, bval = compute_radius_and_bottleneck(centers)
            
            if btype == 'wall':
                # Move the point away from the wall
                i = bidx
                orig = centers[i].copy()
                
                # Determine which wall is closest
                x, y = orig
                if x < -0.5:
                    dx, dy = 0.001, 0
                elif x > 0.5:
                    dx, dy = -0.001, 0
                elif y < -0.5:
                    dx, dy = 0, 0.001
                else:
                    dx, dy = 0, -0.001
                
                # Try multiple directions away from the wall
                best_pos = orig
                best_r = r
                
                for step in [0.0005, 0.001, 0.002, 0.005]:
                    for angle in range(0, 360, 30):
                        rad = math.radians(angle)
                        dx = step * math.cos(rad)
                        dy = step * math.sin(rad)
                        
                        new_center = orig + np.array([dx, dy])
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        
                        centers[i] = new_center
                        new_r, _, _, _ = compute_radius_and_bottleneck(centers)
                        
                        if new_r > best_r + 1e-10:
                            best_r = new_r
                            best_pos = new_center.copy()
                
                centers[i] = best_pos
                
            elif btype == 'pair':
                # Move one of the pair apart
                i, j = bidx
                orig_i = centers[i].copy()
                orig_j = centers[j].copy()
                
                # Try moving point i
                best_pos_i = orig_i
                best_pos_j = orig_j
                best_r = r
                
                for step in [0.0005, 0.001, 0.002, 0.005]:
                    for angle in range(0, 360, 30):
                        rad = math.radians(angle)
                        dx = step * math.cos(rad)
                        dy = step * math.sin(rad)
                        
                        # Try moving i in this direction
                        new_center_i = orig_i + np.array([dx, dy])
                        if -0.5 <= new_center_i[0] <= 0.5 and -0.5 <= new_center_i[1] <= 0.5:
                            centers[i] = new_center_i
                            new_r, _, _, _ = compute_radius_and_bottleneck(centers)
                            if new_r > best_r + 1e-10:
                                best_r = new_r
                                best_pos_i = new_center_i.copy()
                            centers[i] = orig_i
                        
                        # Try moving j in opposite direction
                        new_center_j = orig_j - np.array([dx, dy])
                        if -0.5 <= new_center_j[0] <= 0.5 and -0.5 <= new_center_j[1] <= 0.5:
                            centers[j] = new_center_j
                            new_r, _, _, _ = compute_radius_and_bottleneck(centers)
                            if new_r > best_r + 1e-10:
                                best_r = new_r
                                best_pos_j = new_center_j.copy()
                            centers[j] = orig_j
                
                centers[i] = best_pos_i
                centers[j] = best_pos_j
            
            # Check if we've made progress
            new_r, _, _, _ = compute_radius_and_bottleneck(centers)
            if abs(new_r - r) < 1e-12:
                # No improvement, try a random perturbation to escape local minimum
                if iteration % 10 == 0:
                    idx = np.random.randint(N)
                    perturbation = np.random.randn(2) * 0.001
                    new_center = centers[idx] + perturbation
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[idx] = new_center
                else:
                    break
        
        return centers
    
    def simulated_annealing_polish(centers, start_temp=0.001, end_temp=0.0001, max_iter=500):
        """Simulated annealing to escape local minima."""
        centers = centers.copy()
        current_r, _, _, _ = compute_radius_and_bottleneck(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Annealing schedule
            temp = start_temp * (end_temp / start_temp) ** (iteration / max_iter)
            
            # Pick a random point to move
            i = np.random.randint(N)
            orig = centers[i].copy()
            
            # Random perturbation
            perturbation = np.random.randn(2) * temp
            new_center = orig + perturbation
            
            # Check bounds
            if new_center[0] < -0.5 or new_center[0] > 0.5 or new_center[1] < -0.5 or new_center[1] > 0.5:
                continue
            
            centers[i] = new_center
            new_r, _, _, _ = compute_radius_and_bottleneck(centers)
            
            # Accept if better or with probability exp((new_r - current_r) / temp)
            if new_r > current_r:
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
            else:
                # Metropolis acceptance
                delta = new_r - current_r
                if delta < 0 and np.random.rand() < np.exp(delta / (temp + 1e-10)):
                    current_r = new_r
                else:
                    centers[i] = orig
        
        return best_centers, best_r
    
    # Generate initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try hexagonal layouts
    for n_rows in range(12, 18):
        if time.time() > deadline - 30:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.92, 0.95, 0.98, 1.0, 1.02, 1.05]:
            pts = hexagonal_initialization(n_rows, n_cols, d_scale)
            if pts is not None:
                r, _, _, _ = compute_radius_and_bottleneck(pts)
                if r > best_radius:
                    best_radius = r
                    best_centers = pts.copy()
    
    # Try square grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    r, _, _, _ = compute_radius_and_bottleneck(grid)
    if r > best_radius:
        best_radius = r
        best_centers = grid.copy()
    
    # Apply bottleneck elimination
    if best_centers is not None:
        refined = bottleneck_elimination(best_centers, max_iterations=150)
        r, _, _, _ = compute_radius_and_bottleneck(refined)
        if r > best_radius:
            best_radius = r
            best_centers = refined.copy()
    
    # Apply simulated annealing polish
    if best_centers is not None and time.time() < deadline - 20:
        sa_centers, sa_r = simulated_annealing_polish(best_centers, start_temp=0.0005, end_temp=0.00005, max_iter=300)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5:
        centers = best_centers.copy()
        for step in [0.0002, 0.0001, 0.00005]:
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r, _, _, _ = compute_radius_and_bottleneck(centers)
                
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_center = orig + np.array([dx, dy])
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                        centers[i] = new_center
                        r, _, _, _ = compute_radius_and_bottleneck(centers)
                        if r > best_r + 1e-12:
                            best_r = r
                            best_pos = new_center.copy()
                
                centers[i] = best_pos
        
        r, _, _, _ = compute_radius_and_bottleneck(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
