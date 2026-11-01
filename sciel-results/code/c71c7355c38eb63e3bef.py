import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Create a dense spherical mesh to approximate the sphere
    # Using a grid in (theta, phi)
    n_grid = 200
    theta = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(theta, phi_grid)
    
    # Convert to Cartesian
    mesh_x = np.sin(THETA) * np.cos(PHI)
    mesh_y = np.sin(THETA) * np.sin(PHI)
    mesh_z = np.cos(THETA)
    mesh_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()])
    
    # Normalize mesh points (they should be unit, but just in case)
    mesh_points = mesh_points / np.linalg.norm(mesh_points, axis=1, keepdims=True)
    
    # Function to compute covering radius on the mesh
    def compute_covering_radius(pts):
        # pts: (N, 3)
        # mesh_points: (M, 3)
        # Compute dot products: (M, N)
        dots = mesh_points @ pts.T
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # Angles
        angles = np.arccos(dots)
        # Min angle for each mesh point
        min_angles = np.min(angles, axis=1)
        # Max min angle is the covering radius (in radians)
        return np.max(min_angles)
    
    current_radius = compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Optimization Loop
    # We will use a simple gradient-free descent:
    # Find the worst point on the mesh, and move the nearest configuration points towards it.
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Find the worst mesh point
        dots = mesh_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        max_idx = np.argmax(min_angles)
        worst_point = mesh_points[max_idx]
        current_radius = min_angles[max_idx]
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Reduce step size
            step_size *= 0.5
            no_improve_count = 0
            if step_size < 1e-6:
                break
                
        # Find the K nearest points to the worst_point
        # Distance is 1 - dot
        dists_to_worst = 1.0 - (points @ worst_point)
        K = 3
        nearest_indices = np.argsort(dists_to_worst)[:K]
        
        # Move these points towards worst_point
        # We want to increase the dot product with worst_point
        # p_new = p + step * (worst_point - p) ? 
        # Actually, moving on the sphere:
        # We can use spherical linear interpolation or just vector addition + normalization.
        # Vector addition is simpler and works for small steps.
        
        for idx in nearest_indices:
            # Move towards worst_point
            # Direction is worst_point - points[idx]
            # Project onto tangent plane? 
            # Simple approach: p_new = points[idx] + step_size * worst_point
            # Then normalize.
            # This moves the point towards worst_point.
            
            new_p = points[idx] + step_size * worst_point
            # Normalize
            norm = np.linalg.norm(new_p)
            if norm > 1e-8:
                points[idx] = new_p / norm
        
        # Optional: Check if points are too close to each other (coincident)
        # If they are, we might need to separate them, but for covering,
        # we generally want to spread them. The "move towards worst" might
        # cause clustering. To avoid this, we could add a small repulsive force
        # between very close points.
        
        # Simple repulsion check
        # Compute pairwise distances
        # This is O(N^2) = 130^2 = 16900, very fast.
        # If any distance is < 0.01 rad, push them apart.
        # For now, let's stick to the simple attraction to the gap.
        # If it fails, we can add repulsion.
        
    return best_points
