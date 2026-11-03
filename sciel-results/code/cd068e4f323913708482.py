"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 300 equal circles in the unit disk using a physics-based simulation
    with spatial hashing for efficiency and hexagonal initialization for convergence.
    """
    N = 300
    
    # 1. Hexagonal Initialization
    # Estimate optimal spacing based on area
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate hex grid points
    points = []
    max_idx = int(1.5 / s) + 2
    
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # Select exactly N points closest to the center
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback: Random points if grid is insufficient
        points = np.random.uniform(-1, 1, (N, 2))
        mask = points[:,0]**2 + points[:,1]**2 <= 1
        while mask.sum() < N:
            new_pts = np.random.uniform(-1, 1, (N, 2))
            mask = new_pts[:,0]**2 + new_pts[:,1]**2 <= 1
            points = np.vstack([points, new_pts[mask]])
        points = points[:N]
        
    positions = points
    velocities = np.zeros_like(positions)
    
    # 2. Physics Simulation with Spatial Hashing
    r_sim = 0.053  # Target radius for simulation, close to known optimum
    k_rep = 500.0  # Stronger repulsion for faster convergence
    k_bound = 1000.0
    dt = 0.005
    damping = 0.9
    max_iter = 2000
    
    # Grid parameters for spatial hashing
    cell_size = 2 * r_sim * 1.2  # Cell size slightly larger than interaction range
    grid_offset = 1.0 # Shift coordinates to be positive for indexing
    
    for step in range(max_iter):
        # Build spatial hash grid
        grid = {}
        for i in range(N):
            x, y = positions[i]
            cx = int((x + grid_offset) / cell_size)
            cy = int((y + grid_offset) / cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
            
        forces = np.zeros_like(positions)
        
        # Calculate inter-particle forces
        for i in range(N):
            x, y = positions[i]
            cx = int((x + grid_offset) / cell_size)
            cy = int((y + grid_offset) / cell_size)
            
            # Check 3x3 neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i == j:
                                continue
                            
                            dx_pos = positions[i][0] - positions[j][0]
                            dy_pos = positions[i][1] - positions[j][1]
                            d_sq = dx_pos*dx_pos + dy_pos*dy_pos
                            
                            # Interaction range
                            if d_sq < (2 * r_sim)**2 and d_sq > 1e-12:
                                d = np.sqrt(d_sq)
                                # Repulsive force (Hooke's law)
                                f_mag = k_rep * (2 * r_sim - d)
                                f_x = f_mag * dx_pos / d
                                f_y = f_mag * dy_pos / d
                                forces[i][0] += f_x
                                forces[i][1] += f_y
                                
        # Boundary forces
        dist_center = np.linalg.norm(positions, axis=1)
        limit = 1.0 - r_sim
        overflow = np.maximum(0, dist_center - limit)
        safe_dist = np.where(dist_center > 1e-9, dist_center, 1.0)
        boundary_force = -k_bound * overflow / safe_dist
        forces[:, 0] += boundary_force * positions[:, 0]
        forces[:, 1] += boundary_force * positions[:, 1]
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt
        
        # Early stopping check (optional, but safe)
        if step > 100 and np.max(np.linalg.norm(velocities, axis=1)) < 1e-4:
            break
            
    return positions
# EVOLVE-BLOCK-END
