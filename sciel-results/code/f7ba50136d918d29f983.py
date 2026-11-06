import numpy as np

N = 200

def construct():
    # Generate multiple initial configurations
    configs = []
    
    # Config 1: 10x20 Rectangular Grid
    m, n = 10, 20
    xs = np.linspace(-0.5, 0.5, n)
    ys = np.linspace(-0.5, 0.5, m)
    grid = np.array([(x, y) for y in ys for x in xs])
    configs.append(grid)

    # Config 2: 14x14 Grid (196 points) + 4 points
    # We can just use a 15x14 grid (210 points) and drop 10
    # Or better, use a 14x14 grid and add 4 points in the center of the 4 largest "gaps"?
    # Simpler: Use a 15x15 grid (225 points) and drop 25.
    # Let's try 14x14 (196) and add 4 points at specific locations.
    # Actually, let's just use a 14x14 grid and add 4 points randomly in the middle?
    # Better: Use a 10x20 grid and a 14x14 grid and a Hex grid.
    
    # Config 3: Hexagonal Lattice
    # Estimate spacing d such that we get ~200 points
    # Area per point ~ d^2 * sqrt(3) / 2
    # 200 * d^2 * sqrt(3) / 2 ~ 1 => d ~ 0.075
    d = 0.075
    hex_points = []
    # Range of j
    y_max = 0.5
    y_min = -0.5
    # y = j * d * sqrt(3)/2
    # j range: -y_max / (d*sqrt(3)/2) to y_max / (d*sqrt(3)/2)
    j_max = int(np.ceil(y_max / (d * np.sqrt(3) / 2)))
    j_min = int(np.floor(y_min / (d * np.sqrt(3) / 2)))
    
    for j in range(j_min, j_max + 1):
        y = j * d * np.sqrt(3) / 2
        if y < y_min or y > y_max:
            continue
        # x = i * d + (j % 2) * d / 2
        # i range: -x_max / d to x_max / d
        x_offset = (j % 2) * d / 2
        i_max = int(np.ceil((0.5 - x_offset) / d))
        i_min = int(np.floor((-0.5 - x_offset) / d))
        for i in range(i_min, i_max + 1):
            x = i * d + x_offset
            if x < -0.5 or x > 0.5:
                continue
            hex_points.append([x, y])
            
    hex_points = np.array(hex_points)
    if len(hex_points) > N:
        # Select N points. Keep the ones closest to the center? 
        # Or just the first N? 
        # To maintain symmetry, we should select points that are well-distributed.
        # A simple heuristic: sort by distance to center, take N? No, that clusters in center.
        # Better: Take points in a grid-like order.
        # Let's just take the first N. It might be biased to one corner.
        # Let's center the lattice.
        # The lattice generation above is roughly centered if j_min, j_max are symmetric.
        # Let's just use the first N.
        configs.append(hex_points[:N])
    elif len(hex_points) < N:
        # Add random points in the free space?
        # For now, just use what we have and pad with random points in the center?
        # This might be bad. Let's skip this config if it's too small.
        # Or pad with points from a grid.
        # Let's just use the hex points we have and add random points.
        current = hex_points
        remaining = N - len(current)
        new_points = np.random.uniform(-0.5, 0.5, size=(remaining, 2))
        configs.append(np.vstack([current, new_points]))
    else:
        configs.append(hex_points)

    best_score = -1
    best_points = configs[0]
    
    for points in configs:
        # Ensure shape is (N, 2)
        if len(points) != N:
            continue
        # Run local search
        optimized = local_search(points)
        score = calculate_score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized
            
    return best_points

def calculate_score(points):
    # Calculate the radius
    # min distance between centers / 2
    # min distance to boundary
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_center_dist = np.min(dist, axis=1)
    
    # Distance to boundary
    # Square [-0.5, 0.5]
    # dist to x=-0.5 is x + 0.5
    # dist to x=0.5 is 0.5 - x
    # dist to y=-0.5 is y + 0.5
    # dist to y=0.5 is 0.5 - y
    dx_min = points[:, 0] + 0.5
    dx_max = 0.5 - points[:, 0]
    dy_min = points[:, 1] + 0.5
    dy_max = 0.5 - points[:, 1]
    min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
    
    # The radius is limited by min(min_center_dist/2, min_boundary_dist)
    # We want to maximize this.
    # Note: The task says r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    # My min_center_dist is the min over j for each i. The global min is min(min_center_dist).
    global_min_center = np.min(min_center_dist)
    global_min_boundary = np.min(min_boundary_dist)
    
    r = min(global_min_center / 2.0, global_min_boundary)
    return r

def local_search(points, iterations=1000):
    points = points.copy()
    N = len(points)
    step_size = 0.01
    
    for _ in range(iterations):
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find nearest neighbor index for each point
        nn_idx = np.argmin(dist, axis=1)
        nn_dist = dist[np.arange(N), nn_idx]
        
        # Distance to boundary
        dx_min = points[:, 0] + 0.5
        dx_max = 0.5 - points[:, 0]
        dy_min = points[:, 1] + 0.5
        dy_max = 0.5 - points[:, 1]
        min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
        
        # Determine force direction
        # If nn_dist < 2 * min_boundary_dist, repel from NN
        # Else, repel from boundary
        
        # We want to maximize the minimum distance.
        # This is equivalent to pushing points apart.
        
        # Update velocities/positions
        updates = np.zeros_like(points)
        
        for i in range(N):
            if nn_dist[i] < 2 * min_boundary_dist[i]:
                # Repel from nearest neighbor
                j = nn_idx[i]
                vec = points[i] - points[j]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    updates[i] += vec / norm
            else:
                # Repel from boundary
                # Find which boundary is closest
                if dx_min[i] == min_boundary_dist[i]:
                    updates[i][0] += 1.0 # Move right
                elif dx_max[i] == min_boundary_dist[i]:
                    updates[i][0] -= 1.0 # Move left
                elif dy_min[i] == min_boundary_dist[i]:
                    updates[i][1] += 1.0 # Move up
                elif dy_max[i] == min_boundary_dist[i]:
                    updates[i][1] -= 1.0 # Move down
        
        # Normalize updates to unit vectors?
        # The magnitudes are 1.
        
        # Apply updates
        points += step_size * updates
        
        # Clip to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.995
        
    return points
