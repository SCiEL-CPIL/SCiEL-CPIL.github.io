import numpy as np
import time

N = 200

def construct():
    """
    Packs N equal circles in a unit square [-0.5, 0.5]^2 to maximize radius.
    Uses Hexagonal initialization followed by Simulated Annealing and Greedy Local Search.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def calc_radius(centers):
        """Calculate the maximum possible radius for the given centers."""
        # Distance to boundary
        # centers shape (N, 2)
        # dist to left/right: 0.5 - |x|
        # dist to up/down: 0.5 - |y|
        min_boundary = np.min(np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])))
        
        if min_boundary < 0:
            return 0.0

        # Distance between circles
        # Vectorized pairwise distance
        # diff shape (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dists_sq))
        
        return min(min_boundary, min_pair_dist / 2.0)

    def hex_init():
        """
        Generate a hexagonal lattice initialization.
        """
        # Approximate number of rows/cols
        # Area per hex circle in packing ~ r^2 * 2 * sqrt(3)
        # For N=200, r ~ 0.036. Area ~ 200 * 0.036^2 * 3.46 ~ 0.087
        # Square area 1.
        # Let's just generate a dense hex grid and pick the best N.
        
        # Generate grid points
        # Hex spacing: dx = 2*r, dy = r*sqrt(3)
        # We don't know r yet, but we can estimate based on N.
        # r_est = 0.5 / sqrt(N) * 0.9 (rough)
        r_est = 0.035
        dx = 2 * r_est
        dy = r_est * np.sqrt(3)
        
        # Number of points in x and y to cover [-0.5, 0.5]
        nx = int(1.0 / dx) + 3
        ny = int(1.0 / dy) + 3
        
        points = []
        for j in range(-2, ny + 2):
            for i in range(-2, nx + 2):
                x = i * dx
                y = j * dy
                if j % 2 != 0:
                    x += dx / 2.0
                points.append([x, y])
                
        points = np.array(points)
        
        # Filter points inside the square (with a small margin to allow for movement)
        mask = (np.abs(points[:, 0]) <= 0.5) & (np.abs(points[:, 1]) <= 0.5)
        points = points[mask]
        
        # If we have more than N points, select N points that maximize the minimum distance?
        # That's expensive. Instead, we can just take the first N or random N.
        # Better: Take a subset that is roughly uniform.
        # Or simply: if len(points) >= N, randomly select N? 
        # Random selection from a dense hex grid is a good start.
        if len(points) >= N:
            indices = np.random.choice(len(points), N, replace=False)
            centers = points[indices].copy()
        else:
            # If not enough, pad with random points
            centers = np.vstack([points, np.random.uniform(-0.5, 0.5, (N - len(points), 2))])
            
        # Jitter to break symmetry
        centers += np.random.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, -0.5, 0.5)
        
        return centers

    # 1. Initialization
    np.random.seed(42)
    centers = hex_init()
    current_radius = calc_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # 2. Optimization Loop
    # We will use a hybrid of Simulated Annealing (for global moves) and Greedy Local Search (for polishing)
    
    # Parameters for Greedy Search
    # Directions to try for each point
    # We will try moving towards the "free space"
    
    while time.time() - start_time < time_limit:
        
        # --- Phase A: Greedy Local Search ---
        # Try to improve by moving each point to a better local position
        improved_in_pass = False
        
        for i in range(N):
            # Find the current bottleneck for point i
            # 1. Distance to boundary
            dist_bound = np.min(0.5 - np.abs(centers[i]))
            
            # 2. Distance to other points
            dists = np.sum((centers - centers[i])**2, axis=1)
            dists[i] = np.inf
            min_dist_sq = np.min(dists)
            min_dist = np.sqrt(min_dist_sq)
            
            # The limiting radius for this point is min(dist_bound, min_dist/2)
            # But the global radius is min over all points.
            # To improve the global radius, we should move the point that is most "constrained"
            # i.e., the one with the smallest min(dist_bound, min_dist/2)
            
            # For simplicity, we just try to move point i to increase its local clearance.
            
            # Candidate moves:
            # 1. Move towards the center (if boundary is limiting)
            # 2. Move away from the nearest neighbor (if neighbor is limiting)
            # 3. Random small steps
            
            best_local_move = None
            best_local_radius = current_radius
            
            # Find nearest neighbor index
            j = np.argmin(dists)
            
            # Direction away from j
            dir_away = centers[i] - centers[j]
            norm_away = np.linalg.norm(dir_away)
            if norm_away > 1e-9:
                dir_away /= norm_away
            
            # Direction towards center
            dir_center = -centers[i]
            norm_center = np.linalg.norm(dir_center)
            if norm_center > 1e-9:
                dir_center /= norm_center
            
            # Try moving along dir_away and dir_center and some random directions
            directions = [dir_away, dir_center]
            # Add 4 random directions
            angles = np.random.uniform(0, 2*np.pi, 4)
            for a in angles:
                directions.append([np.cos(a), np.sin(a)])
            
            # Try steps
            max_step = min(min_dist, dist_bound) * 0.5
            if max_step < 1e-6:
                max_step = 0.01
                
            steps = [0.1, 0.2, 0.5, 0.8, 1.0]
            
            for d in directions:
                for s_factor in steps:
                    step_size = max_step * s_factor
                    new_pos = centers[i] + step_size * d
                    
                    # Check bounds
                    if np.any(np.abs(new_pos) > 0.5):
                        continue
                    
                    # Calculate new radius
                    temp_centers = centers.copy()
                    temp_centers[i] = new_pos
                    new_r = calc_radius(temp_centers)
                    
                    if new_r > best_local_radius + 1e-12:
                        best_local_radius = new_r
                        best_local_move = new_pos
            
            if best_local_move is not None:
                centers[i] = best_local_move
                current_radius = best_local_radius
                improved_in_pass = True
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
                
                # Early break if we found a significant improvement? 
                # No, continue to polish other points.
        
        # --- Phase B: Shake if stuck ---
        if not improved_in_pass:
            # No improvement in a full pass. Shake to escape local minimum.
            # Randomly perturb a few points
            num_shake = 5
            idxs = np.random.choice(N, num_shake, replace=False)
            for idx in idxs:
                centers[idx] += np.random.normal(0, 0.02, 2)
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
            
            current_radius = calc_radius(centers)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                
    return best_centers
