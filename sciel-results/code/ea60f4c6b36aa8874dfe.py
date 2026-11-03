"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Compute pairwise distances
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)
    
    def compute_radius_and_bottleneck(centers):
        """Compute radius and identify the bottleneck."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        min_wall = wall_dists.flat[min_wall_idx]
        
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_idx = np.argmin(dist_matrix)
        min_pair = dist_matrix.flat[min_pair_idx]
        
        if min_wall < min_pair / 2.0:
            bottleneck_type = 'wall'
            bottleneck_idx = min_wall_idx // 2
            bottleneck_value = min_wall
        else:
            i, j = np.unravel_index(min_pair_idx, (n, n))
            bottleneck_type = 'pair'
            bottleneck_idx = (i, j)
            bottleneck_value = min_pair / 2.0
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r), bottleneck_type, bottleneck_idx, bottleneck_value
    
    def hexagonal_packing(n_rows, n_cols, offset_rows=True):
        """Create hexagonal packing layout."""
        # Calculate spacing to fit in unit square
        # For hexagonal packing, horizontal spacing = d, vertical spacing = d * sqrt(3)/2
        # Total width = (n_cols - 1) * d + d = n_cols * d (if centered)
        # Actually, let's compute the optimal d
        
        # The width needed is (n_cols - 1) * d + 2*r where r = d/2
        # So width = (n_cols - 1) * d + d = n_cols * d
        # We need n_cols * d <= 1, so d <= 1/n_cols
        # But we also need to fit vertically
        
        # Let's compute d from the constraint that fits in the square
        # Width constraint: (n_cols - 1) * d + d <= 1 => n_cols * d <= 1 => d <= 1/n_cols
        # Height constraint: (n_rows - 1) * d * sqrt(3)/2 + d <= 1
        # => d * ((n_rows - 1) * sqrt(3)/2 + 1) <= 1
        # => d <= 1 / ((n_rows - 1) * sqrt(3)/2 + 1)
        
        d_from_width = 1.0 / n_cols
        d_from_height = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_from_width, d_from_height)
        
        # Now center the packing in the unit square
        total_width = (n_cols - 1) * d + d  # = n_cols * d
        total_height = (n_rows - 1) * d * math.sqrt(3) / 2 + d
        
        x_offset = -0.5 + total_width / 2 - d / 2
        y_offset = -0.5 + total_height / 2 - d / 2
        
        pts = []
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if offset_rows and row % 2 == 1:
                # Offset this row
                n_in_row = n_cols - 1
                x_start = x_offset + d / 2
            else:
                n_in_row = n_cols
                x_start = x_offset
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with points at the edges or just return what we have
            # This shouldn't happen if we chose n_rows, n_cols correctly
            pass
        
        return pts
    
    def greedy_select_from_lattice(lattice_points, n_select):
        """Greedy selection from lattice points to maximize minimum distance."""
        selected = []
        remaining = list(range(len(lattice_points)))
        
        # Start with the point that is farthest from the boundary
        # Or just start with a corner-ish point
        # Actually, let's start with the point that maximizes distance to boundary
        # and then greedily add points
        
        # First, select the point that is best positioned
        # We'll use a different strategy: start from the center and expand
        
        # Actually, let's just do pure greedy:
        # At each step, select the point that maximizes the minimum distance to already selected points
        # and to the boundary
        
        for _ in range(n_select):
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining:
                pt = lattice_points[idx]
                
                # Distance to boundary
                dist_to_boundary = min(0.5 - abs(pt[0]), 0.5 - abs(pt[1]))
                
                # Distance to selected points
                if selected:
                    selected_arr = np.array(selected)
                    dists_to_selected = np.min(cdist(pt.reshape(1, 2), selected_arr)[0])
                    min_dist = min(dist_to_boundary, dists_to_selected)
                else:
                    min_dist = dist_to_boundary
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(lattice_points[best_idx])
            remaining.remove(best_idx)
        
        return np.array(selected)
    
    def create_fine_lattice(spacing):
        """Create a fine hexagonal lattice."""
        pts = []
        # Create hexagonal lattice with given spacing
        row_height = spacing * math.sqrt(3) / 2
        
        # Determine how many rows and columns we need
        # We want to cover [-0.5, 0.5] x [-0.5, 0.5]
        
        y = -0.5
        row = 0
        while y <= 0.5:
            x = -0.5
            if row % 2 == 1:
                x = -0.5 + spacing / 2
            while x <= 0.5:
                pts.append([x, y])
                x += spacing
            y += row_height
            row += 1
        
        return np.array(pts)
    
    def local_optimization(centers, max_iterations=100):
        """Local optimization using steepest descent on the bottleneck."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            r, btype, bidx, bval = compute_radius_and_bottleneck(centers)
            
            if btype == 'wall':
                i = bidx
                orig = centers[i].copy()
                x, y = orig
                
                # Determine which wall is closest
                if x < -0.5:
                    direction = np.array([1, 0])
                elif x > 0.5:
                    direction = np.array([-1, 0])
                elif y < -0.5:
                    direction = np.array([0, 1])
                else:
                    direction = np.array([0, -1])
                
                # Try moving in the direction away from the wall
                best_pos = orig
                best_r = r
                
                for step in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
                    new_center = orig + direction * step
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                
                centers[i] = best_pos
                
            elif btype == 'pair':
                i, j = bidx
                orig_i = centers[i].copy()
                orig_j = centers[j].copy()
                
                # Direction from j to i
                direction = (orig_i - orig_j)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                else:
                    direction = np.array([1, 0])
                
                best_pos_i = orig_i
                best_pos_j = orig_j
                best_r = r
                
                for step in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
                    # Move i away from j
                    new_i = orig_i + direction * step
                    if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5:
                        centers[i] = new_i
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_i = new_i.copy()
                    centers[i] = orig_i
                    
                    # Move j away from i
                    new_j = orig_j - direction * step
                    if -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                        centers[j] = new_j
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_j = new_j.copy()
                    centers[j] = orig_j
                
                centers[i] = best_pos_i
                centers[j] = best_pos_j
            
            # Check for convergence
            new_r = compute_radius(centers)
            if abs(new_r - current_r) < 1e-14:
                if iteration % 5 == 0:
                    # Random perturbation
                    idx = np.random.randint(N)
                    perturbation = np.random.randn(2) * 0.001
                    new_center = centers[idx] + perturbation
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[idx] = new_center
                else:
                    break
            
            current_r = new_r
        
        return centers
    
    def simulated_annealing(centers, start_temp=0.002, end_temp=0.0001, max_iter=1000):
        """Simulated annealing optimization."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Annealing schedule
            temp = start_temp * (end_temp / start_temp) ** (iteration / max_iter)
            
            # Pick a random point
            i = np.random.randint(N)
            orig = centers[i].copy()
            
            # Random perturbation
            perturbation = np.random.randn(2) * temp
            new_center = orig + perturbation
            
            # Check bounds
            if new_center[0] < -0.5 or new_center[0] > 0.5 or new_center[1] < -0.5 or new_center[1] > 0.5:
                continue
            
            centers[i] = new_center
            new_r = compute_radius(centers)
            
            # Accept if better or with Metropolis probability
            if new_r >= current_r:
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
            else:
                delta = new_r - current_r
                if np.random.rand() < np.exp(delta / (temp + 1e-10)):
                    current_r = new_r
                else:
                    centers[i] = orig
        
        return best_centers, best_r
    
    # Generate initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try different hexagonal layouts
    for n_rows in range(12, 20):
        if time.time() > deadline - 40:
            break
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_packing(n_rows, n_cols)
        if pts is not None and len(pts) == N:
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()
    
    # Try square grid
    m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        grid = grid[:N]
        r = compute_radius(grid)
        if r > best_radius:
            best_radius = r
            best_centers = grid.copy()
    
    # Try greedy selection from fine lattice
    if time.time() < deadline - 30:
        for spacing in [0.02, 0.025, 0.03]:
            if time.time() > deadline - 30:
                break
            lattice = create_fine_lattice(spacing)
            if len(lattice) >= N:
                greedy_pts = greedy_select_from_lattice(lattice, N)
                if len(greedy_pts) == N:
                    r = compute_radius(greedy_pts)
                    if r > best_radius:
                        best_radius = r
                        best_centers = greedy_pts.copy()
    
    # Apply local optimization
    if best_centers is not None:
        refined = local_optimization(best_centers, max_iterations=200)
        r = compute_radius(refined)
        if r > best_radius:
            best_radius = r
            best_centers = refined.copy()
    
    # Apply simulated annealing
    if best_centers is not None and time.time() < deadline - 20:
        sa_centers, sa_r = simulated_annealing(best_centers, start_temp=0.001, end_temp=0.00005, max_iter=500)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
    
    # Final polish
    if best_centers is not None and time.time() < deadline - 5:
        centers = best_centers.copy()
        for step in [0.0002, 0.0001, 0.00005]:
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r = compute_radius(centers)
                
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
                        r = compute_radius(centers)
                        if r > best_r + 1e-12:
                            best_r = r
                            best_pos = new_center.copy()
                
                centers[i] = best_pos
        
        r = compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
