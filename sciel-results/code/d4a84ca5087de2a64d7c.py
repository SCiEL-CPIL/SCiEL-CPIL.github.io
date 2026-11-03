"""Improved program using Voronoi-based gradient ascent."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _voronoi_optimize(centers, time_limit=30.0):
    """
    Optimize packing using Voronoi diagram gradients.
    Moves centers away from their closest Voronoi boundary or square wall.
    """
    centers = centers.copy()
    best_r = _calculate_true_radius(centers)
    step_size = 0.005
    start_time = time.time()
    no_improve_count = 0
    
    # Pre-allocate for performance
    n = len(centers)
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Compute Voronoi Diagram
        try:
            vor = Voronoi(centers)
        except Exception:
            # Fallback if Voronoi fails (e.g., degenerate points)
            centers += np.random.normal(0, 0.001, (n, 2))
            continue

        # 2. Determine constraints and gradients for each center
        # We need to find the closest feature (Voronoi edge or square wall) for each center.
        # And the direction to move away from it.
        
        gradients = np.zeros((n, 2))
        radii = np.full(n, 0.5) # Initialize with max possible radius
        
        for i in range(n):
            c = centers[i]
            min_dist = 0.5 # Distance to wall is at most 0.5
            min_dir = np.array([0.0, 0.0])
            
            # Check square walls
            # Left wall: x = -0.5
            d_left = c[0] - (-0.5)
            if d_left < min_dist:
                min_dist = d_left
                min_dir = np.array([-1.0, 0.0]) # Move left? No, move away from wall. Wall is at left, so move Right?
                # Wait. If c[0] is close to -0.5, the wall is to the left. To increase distance, we move Right (+x).
                # So direction is +x.
                min_dir = np.array([1.0, 0.0])
            
            # Right wall: x = 0.5
            d_right = 0.5 - c[0]
            if d_right < min_dist:
                min_dist = d_right
                min_dir = np.array([-1.0, 0.0]) # Move Left (-x)
                
            # Bottom wall: y = -0.5
            d_bottom = c[1] - (-0.5)
            if d_bottom < min_dist:
                min_dist = d_bottom
                min_dir = np.array([0.0, 1.0]) # Move Up (+y)
                
            # Top wall: y = 0.5
            d_top = 0.5 - c[1]
            if d_top < min_dist:
                min_dist = d_top
                min_dir = np.array([0.0, -1.0]) # Move Down (-y)

            # Check Voronoi edges
            # The Voronoi cell for point i is defined by its ridges.
            # vor.ridge_points gives indices of the two points defining each ridge.
            # We only care about ridges that are part of point i's cell.
            # vor.regions gives the list of vertex indices for each cell.
            # A ridge is an edge between two vertices in the cell.
            
            # Efficient way: Iterate over all ridges and check if i is one of the points.
            # Since N=50, this is fine.
            # vor.ridge_points has shape (M, 2)
            # vor.ridge_vertices has shape (M, 2)
            
            for m in range(len(vor.ridge_points)):
                p1_idx, p2_idx = vor.ridge_points[m]
                if p1_idx != i and p2_idx != i:
                    continue
                
                v1_idx, v2_idx = vor.ridge_vertices[m]
                if v1_idx == -1 or v2_idx == -1:
                    # Infinite ridge, handled by wall checks above? 
                    # Actually, infinite ridges go to infinity. 
                    # For points on the convex hull, the Voronoi cell is unbounded.
                    # The "wall" of the square will be the limiting factor.
                    # So we can skip infinite ridges for the "closest feature" check 
                    # unless the point is far from the square wall.
                    # But our wall check already covers the square boundary.
                    # The Voronoi edge might be closer than the square wall?
                    # If the ridge is infinite, it goes from a finite vertex to infinity.
                    # The segment from the vertex to the point c is the part of the boundary.
                    # We should check the distance to the finite vertex and the ray.
                    # However, for simplicity and robustness, we often just rely on the 
                    # finite vertices and the square walls. 
                    # Let's check the distance to the finite vertex if one exists.
                    if v1_idx != -1:
                        v = vor.vertices[v1_idx]
                        d = np.linalg.norm(c - v)
                        if d < min_dist:
                            min_dist = d
                            min_dir = (c - v) / d if d > 1e-9 else np.array([0,0])
                    if v2_idx != -1:
                        v = vor.vertices[v2_idx]
                        d = np.linalg.norm(c - v)
                        if d < min_dist:
                            min_dist = d
                            min_dir = (c - v) / d if d > 1e-9 else np.array([0,0])
                    continue

                # Finite ridge
                v1 = vor.vertices[v1_idx]
                v2 = vor.vertices[v2_idx]
                
                # Find closest point on segment v1-v2 to c
                # Project c onto line v1-v2
                v12 = v2 - v1
                t = np.dot(c - v1, v12) / np.dot(v12, v12)
                t = np.clip(t, 0.0, 1.0)
                closest_point = v1 + t * v12
                
                d = np.linalg.norm(c - closest_point)
                if d < min_dist:
                    min_dist = d
                    if d > 1e-9:
                        min_dir = (c - closest_point) / d
                    else:
                        min_dir = np.array([0.0, 0.0])

            radii[i] = min_dist
            gradients[i] = min_dir

        # 3. Identify active centers (those with small radius)
        min_r = np.min(radii)
        # We want to increase the global minimum radius.
        # Focus on centers with radius close to the minimum.
        threshold = min_r + 1e-4
        active_mask = radii < threshold
        
        if not np.any(active_mask):
            # Should not happen
            break
            
        active_indices = np.where(active_mask)[0]
        
        # 4. Calculate move
        # Average the gradients of active centers? 
        # Or move each active center individually?
        # Moving individually is safer to avoid collisions.
        # But we want to maximize the *minimum*. 
        # Let's move all active centers in the direction of their gradient.
        
        new_centers = centers.copy()
        for i in active_indices:
            new_centers[i] += step_size * gradients[i]
            
        # Clip to square
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Check improvement
        new_r = _calculate_true_radius(new_centers)
        
        if new_r > best_r + 1e-12:
            centers = new_centers
            best_r = new_r
            no_improve_count = 0
            # Optionally adjust step size
            # step_size = min(step_size * 1.1, 0.05)
        else:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.9
            if no_improve_count > 100:
                # Add noise to escape local minima
                centers += np.random.normal(0, 0.01, (n, 2))
                centers = np.clip(centers, -0.5, 0.5)
                no_improve_count = 0
                step_size = 0.005
            if step_size < 1e-6:
                break

    return centers, best_r

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14 
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0: 
            break
            
        centers, score = _voronoi_optimize(init, time_limit=15.0)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
