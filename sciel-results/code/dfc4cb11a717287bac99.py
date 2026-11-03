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

    def get_active_constraints(centers, r):
        """
        Identify active constraints for radius r.
        Returns:
        - pair_constraints: list of (i, j) where dist(i, j) <= 2*r + eps
        - wall_constraints: list of i where wall_dist(i) <= r + eps
        """
        eps = 1e-9
        wall_dists = 0.5 - np.abs(centers)
        wall_active = np.where(wall_dists <= r + eps)[0]
        
        tree = cKDTree(centers)
        # Query neighbors within 2*r
        pairs = tree.query_pairs(2 * r + eps, output_type='ndarray')
        
        return pairs, wall_active

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

    def deterministic_local_search(centers, max_iterations=500):
        """
        Deterministic Constraint-Based Local Search.
        1. Compute current radius r.
        2. Identify active constraints (tight pairs and walls).
        3. For each active constraint, compute a 'push' direction that increases the slack.
        4. Apply a small step in the aggregated push direction for the most critical points.
        5. Repeat until no improvement or time limit.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # Parameters
        step_size = 0.001
        min_step_size = 1e-6
        improvement_threshold = 1e-12
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # 1. Compute active constraints
            pairs, wall_active = get_active_constraints(centers, current_r)
            
            if len(pairs) == 0 and len(wall_active) == 0:
                break
                
            # 2. Compute push vectors for each point
            # We want to move points to increase the minimum distance.
            # For a pair (i, j), point i should move away from j, and j away from i.
            # For a wall, point i should move away from the wall.
            
            push = np.zeros_like(centers)
            
            # Handle wall constraints
            for i in wall_active:
                # Determine which wall is closest
                dists_to_walls = np.array([
                    centers[i][0] + 0.5, # Left
                    0.5 - centers[i][0], # Right
                    centers[i][1] + 0.5, # Bottom
                    0.5 - centers[i][1]  # Top
                ])
                min_wall_dist = np.min(dists_to_walls)
                
                if min_wall_dist <= current_r + 1e-9:
                    # Push away from the closest wall
                    if dists_to_walls[0] == min_wall_dist:
                        push[i][0] += 1.0
                    elif dists_to_walls[1] == min_wall_dist:
                        push[i][0] -= 1.0
                    elif dists_to_walls[2] == min_wall_dist:
                        push[i][1] += 1.0
                    elif dists_to_walls[3] == min_wall_dist:
                        push[i][1] -= 1.0
            
            # Handle pair constraints
            for i, j in pairs:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Random nudge if coincident
                    push[i] += np.random.randn(2)
                    push[j] -= np.random.randn(2)
                    continue
                    
                # Unit vector from j to i
                u = diff / dist
                
                # i should move in direction u (away from j)
                # j should move in direction -u (away from i)
                # Weight by how tight the constraint is? 
                # For simplicity, equal weight.
                push[i] += u
                push[j] -= u
            
            # Normalize push vectors to prevent huge steps
            norms = np.linalg.norm(push, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            push = push / norms
            
            # 3. Apply step
            # We only move points that have a non-zero push
            active_indices = np.where(np.linalg.norm(push, axis=1) > 1e-6)[0]
            if len(active_indices) == 0:
                break
                
            new_centers = centers.copy()
            new_centers[active_indices] += push[active_indices] * step_size
            
            # Ensure points stay inside the square (hard constraint for validity, 
            # but we allow slight violation if it increases r? No, must be valid.
            # Clip to [-0.5, 0.5] but this might reduce r. 
            # Better: check if new radius is better. If not, revert.
            
            new_r = compute_radius(new_centers)
            
            if new_r > current_r + improvement_threshold:
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                # Reduce step size slightly on success to refine
                step_size *= 0.95
                if step_size < min_step_size:
                    step_size = min_step_size
            else:
                # Failure: increase step size or try a different strategy
                # If we fail to improve, we might be stuck in a local minimum 
                # where all active constraints form a rigid structure.
                # In CDLS, we can try to "break" the rigidity by randomly perturbing 
                # one of the active points slightly.
                if np.random.rand() < 0.1:
                    # Perturb a random active point
                    i = active_indices[np.random.randint(len(active_indices))]
                    centers[i] += np.random.randn(2) * step_size * 0.5
                    # Re-compute r after perturbation? 
                    # No, just let the next iteration handle it. 
                    # But we need to ensure we don't decrease r significantly.
                    perturbed_r = compute_radius(centers)
                    if perturbed_r < current_r - 1e-5:
                        # Revert if too bad
                        centers[i] -= np.random.randn(2) * step_size * 0.5
                        # Increase step size to escape
                        step_size *= 1.1
                    else:
                        current_r = perturbed_r
                        if current_r > best_r:
                            best_r = current_r
                            best_centers = centers.copy()
                else:
                    # Standard failure: increase step size to explore larger moves
                    step_size *= 1.05
                    if step_size > 0.01:
                        step_size = 0.01

        return best_centers, best_r

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
            
        # Deterministic Local Search
        if time.time() < deadline - 10:
            try:
                dls_centers, dls_r = deterministic_local_search(init_pts)
                if dls_r > best_radius:
                    best_radius = dls_r
                    best_centers = dls_centers.copy()
            except Exception:
                pass

    # Final high-precision polish on the best found
    if best_centers is not None and time.time() < deadline - 5:
        # Run DLS again on the best result for fine-tuning
        polished, r_polished = deterministic_local_search(best_centers, max_iterations=200)
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
