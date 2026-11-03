import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if len(centers) < 2:
            return min_wall
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def get_bottlenecks(centers):
        """Identify indices involved in the minimum distance (bottlenecks)."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        wall_bottlenecks = np.where(np.isclose(wall_dists, min_wall, atol=1e-9))[0]
        
        if len(centers) < 2:
            return wall_bottlenecks, np.array([], dtype=int)
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        # Find all pairs with distance close to min_pair
        pair_bottlenecks = set()
        # For efficiency, just take the argmin index and its neighbor
        min_idx = np.argmin(dists[:, 1])
        neighbor_idx = _[min_idx][1]
        pair_bottlenecks.add(min_idx)
        pair_bottlenecks.add(neighbor_idx)
        
        # If wall distance is the bottleneck, add those too
        if min_wall < min_pair / 2.0:
            bottlenecks = np.concatenate([wall_bottlenecks, list(pair_bottlenecks)])
        else:
            bottlenecks = np.array(list(pair_bottlenecks))
            
        return np.unique(bottlenecks), pair_bottlenecks

    def hexagonal_initialization(n_rows, n_cols):
        """Create a hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
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

    def simulated_annealing(centers, max_time):
        """Simulated annealing with bottleneck elimination."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # Temperature parameters
        T = 0.005 * current_r
        T_min = 1e-6
        cooling_rate = 0.995
        
        step_size = 0.01 * current_r
        min_step_size = 1e-5 * current_r
        
        while time.time() < max_time and T > T_min:
            # Identify bottlenecks
            bottlenecks, _ = get_bottlenecks(centers)
            
            if len(bottlenecks) == 0:
                break
                
            # Pick a random bottleneck point to move
            i = bottlenecks[np.random.randint(len(bottlenecks))]
            
            orig_pos = centers[i].copy()
            
            # Generate a random move
            # Bias towards moving away from the nearest neighbor if it's a pair bottleneck
            # But for simplicity, let's do a random Gaussian move
            move = np.random.randn(2) * step_size
            new_pos = orig_pos + move
            
            # Check bounds
            if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                T *= cooling_rate
                step_size *= 0.98
                continue
                
            centers[i] = new_pos
            new_r = compute_radius(centers)
            
            delta_r = new_r - current_r
            
            # Acceptance criterion
            if delta_r > 0 or np.random.rand() < math.exp(delta_r / T):
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                centers[i] = orig_pos
                
            T *= cooling_rate
            step_size *= 0.98
            if step_size < min_step_size:
                step_size = min_step_size
                
            # Periodic restart to escape deep local minima
            if np.random.rand() < 0.01:
                # Randomize a few points
                for _ in range(5):
                    j = np.random.randint(N)
                    centers[j] = orig_pos + np.random.randn(2) * step_size * 2
                    if centers[j][0] < -0.5 or centers[j][0] > 0.5:
                        centers[j][0] = np.clip(centers[j][0], -0.5, 0.5)
                    if centers[j][1] < -0.5 or centers[j][1] > 0.5:
                        centers[j][1] = np.clip(centers[j][1], -0.5, 0.5)
                current_r = compute_radius(centers)
                
        return best_centers, best_r

    def local_search_polish(centers, max_rounds=10):
        """Final high-precision polish using direct local moves."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
                
            improved = False
            tree = cKDTree(centers)
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                orig = centers[i].copy()
                best_pos = orig
                best_r = current_r
                
                # Find nearest neighbors
                dists, idxs = tree.query(orig, k=5)
                
                # Try small moves in 8 directions + random
                step = 0.001 / (round_num + 1)
                directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                directions = [np.array(d) / np.linalg.norm(d) for d in directions]
                for _ in range(4):
                    angle = np.random.rand() * 2 * np.pi
                    directions.append(np.array([math.cos(angle), math.sin(angle)]))
                
                for d in directions:
                    for step_mult in [0.5, 1.0, 2.0, 4.0]:
                        new_center = orig + d * step * step_mult
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                            
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                            improved = True
                            
                centers[i] = best_pos
                
            current_r = best_r
            
            if not improved:
                break
                
        return centers, current_r

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try multiple hexagonal layouts
    configs = []
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            configs.append(pts)
            
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        # Evaluate initial
        r = compute_radius(init_pts)
        if r > best_radius:
            best_radius = r
            best_centers = init_pts.copy()
            
        # Simulated Annealing
        if time.time() < deadline - 15:
            try:
                sa_time_limit = min(deadline - 10, time.time() + 20)
                sa_centers, sa_r = simulated_annealing(init_pts, sa_time_limit)
                if sa_r > best_radius:
                    best_radius = sa_r
                    best_centers = sa_centers.copy()
            except Exception:
                pass
                
        # Final polish on best so far
        if time.time() < deadline - 5:
            polished, r_polished = local_search_polish(best_centers, max_rounds=5)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()

    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 2:
        polished, r_polished = local_search_polish(best_centers, max_rounds=10)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
