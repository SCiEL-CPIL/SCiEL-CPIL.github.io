import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

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
    
    # Config 2: Hexagonal Lattice
    d = 0.075
    hex_points = []
    y_max = 0.5
    y_min = -0.5
    j_max = int(np.ceil(y_max / (d * np.sqrt(3) / 2)))
    j_min = int(np.floor(y_min / (d * np.sqrt(3) / 2)))
    
    for j in range(j_min, j_max + 1):
        y = j * d * np.sqrt(3) / 2
        if y < y_min or y > y_max:
            continue
        x_offset = (j % 2) * d / 2
        i_max = int(np.ceil((0.5 - x_offset) / d))
        i_min = int(np.floor((-0.5 - x_offset) / d))
        for i in range(i_min, i_max + 1):
            x = i * d + x_offset
            if x < -0.5 or x > 0.5:
                continue
            hex_points.append([x, y])
            
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        # Select N points from hex lattice
        # Use a simple approach: take points in order, but ensure we get a good distribution
        # Sort by distance from center and take N closest to maintain symmetry
        center = np.mean(hex_points, axis=0)
        dists = np.linalg.norm(hex_points - center, axis=1)
        # Actually, we want to keep the lattice structure, so let's just take the first N
        # But to be more symmetric, let's try to select points that are well-distributed
        # For now, just take the first N
        configs.append(hex_points[:N])
    else:
        # Pad with random points
        remaining = N - len(hex_points)
        new_points = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        configs.append(np.vstack([hex_points, new_points]))
    
    # Config 3: Random initial configuration
    random_config = np.random.uniform(-0.45, 0.45, size=(N, 2))
    configs.append(random_config)
    
    # Config 4: 14x14 grid (196 points) + 4 extra points
    m2, n2 = 14, 14
    xs2 = np.linspace(-0.5, 0.5, n2)
    ys2 = np.linspace(-0.5, 0.5, m2)
    grid2 = np.array([(x, y) for y in ys2 for x in xs2])
    # Add 4 points at the centers of the 4 quadrants
    extra = np.array([
        [0.0, 0.0],
        [0.125, 0.125],
        [-0.125, 0.125],
        [0.125, -0.125]
    ])
    # Actually, let's just use the 14x14 grid and add 4 points at specific locations
    # Better: use 15x14 = 210, drop 10
    m3, n3 = 15, 14
    xs3 = np.linspace(-0.5, 0.5, n3)
    ys3 = np.linspace(-0.5, 0.5, m3)
    grid3 = np.array([(x, y) for y in ys3 for x in xs3])
    # Drop 10 points - remove from corners to maintain symmetry
    # Remove points that are closest to corners
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    corner_dists = np.min(np.linalg.norm(grid3[:, np.newaxis, :] - corners[np.newaxis, :, :], axis=2), axis=1)
    remove_idx = np.argsort(corner_dists)[:10]
    grid3_opt = np.delete(grid3, remove_idx, axis=0)
    configs.append(grid3_opt)
    
    best_score = -1
    best_points = configs[0]
    
    for points in configs:
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
    
    # Pairwise distances using KDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    # dists[i][0] is distance to self (0), dists[i][1] is distance to nearest neighbor
    min_center_dist = dists[:, 1]
    
    # Distance to boundary
    dx_min = points[:, 0] + 0.5
    dx_max = 0.5 - points[:, 0]
    dy_min = points[:, 1] + 0.5
    dy_max = 0.5 - points[:, 1]
    min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
    
    global_min_center = np.min(min_center_dist)
    global_min_boundary = np.min(min_boundary_dist)
    
    r = min(global_min_center / 2.0, global_min_boundary)
    return r

def local_search(points, max_iterations=500):
    points = points.copy()
    N = len(points)
    
    # Use gradient-based optimization
    def objective(x):
        pts = x.reshape(-1, 2)
        # Clip to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Compute pairwise distances
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_center_dist = dists[:, 1]
        
        # Distance to boundary
        dx_min = pts[:, 0] + 0.5
        dx_max = 0.5 - pts[:, 0]
        dy_min = pts[:, 1] + 0.5
        dy_max = 0.5 - pts[:, 1]
        min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
        
        global_min_center = np.min(min_center_dist)
        global_min_boundary = np.min(min_boundary_dist)
        
        r = min(global_min_center / 2.0, global_min_boundary)
        return -r  # Minimize negative r
    
    # Use Nelder-Mead or similar
    # Actually, let's use a simpler approach: iterative repulsion
    
    step_size = 0.01
    for it in range(max_iterations):
        # Compute pairwise distances
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=2)
        nn_dist = dists[:, 1]
        nn_idx = indices[:, 1]
        
        # Distance to boundary
        dx_min = points[:, 0] + 0.5
        dx_max = 0.5 - points[:, 0]
        dy_min = points[:, 1] + 0.5
        dy_max = 0.5 - points[:, 1]
        min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
        
        # Determine force direction
        updates = np.zeros_like(points)
        
        for i in range(N):
            # Weight based on how constrained this point is
            constraint = min(nn_dist[i] / 2.0, min_boundary_dist[i])
            weight = 1.0 / (constraint + 0.01)
            
            # Contribution from nearest neighbor repulsion
            if nn_dist[i] > 1e-10:
                vec = points[i] - points[nn_idx[i]]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    updates[i] += (vec / norm) * weight
            
            # Contribution from boundary repulsion
            if min_boundary_dist[i] < 0.05:
                if dx_min[i] == min_boundary_dist[i]:
                    updates[i][0] += weight
                elif dx_max[i] == min_boundary_dist[i]:
                    updates[i][0] -= weight
                elif dy_min[i] == min_boundary_dist[i]:
                    updates[i][1] += weight
                elif dy_max[i] == min_boundary_dist[i]:
                    updates[i][1] -= weight
        
        # Normalize updates
        norms = np.linalg.norm(updates, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        updates /= norms
        
        # Apply updates with adaptive step size
        points += step_size * updates
        
        # Clip to square with small margin
        points[:, 0] = np.clip(points[:, 0], -0.499, 0.499)
        points[:, 1] = np.clip(points[:, 1], -0.499, 0.499)
        
        # Decay step size
        step_size *= 0.998
        
        # Add small random perturbation to escape local minima
        if it % 50 == 0 and it > 0:
            points += np.random.normal(0, 0.002, size=points.shape)
            points[:, 0] = np.clip(points[:, 0], -0.499, 0.499)
            points[:, 1] = np.clip(points[:, 1], -0.499, 0.499)
        
        # Check convergence
        if it > 100 and step_size < 0.0001:
            break
    
    return points
