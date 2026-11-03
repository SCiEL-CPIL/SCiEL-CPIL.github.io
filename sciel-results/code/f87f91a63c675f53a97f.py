import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _generate_icosahedral_grid(n_subdiv=4):
    """
    Generate a high-resolution icosahedral grid on the unit sphere.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    
    # Subdivide faces
    points_set = set()
    for v in verts:
        points_set.add(tuple(np.round(v, 10)))
    
    for _ in range(n_subdiv):
        new_faces = []
        for face in faces:
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            
            # Midpoints of edges
            ab = (a + b) / 2
            bc = (b + c) / 2
            ca = (c + a) / 2
            
            # Normalize
            ab /= np.linalg.norm(ab)
            bc /= np.linalg.norm(bc)
            ca /= np.linalg.norm(ca)
            
            # Add new vertices
            for v in [ab, bc, ca]:
                points_set.add(tuple(np.round(v, 10)))
            
            # New faces
            new_faces.append([a, ab, ca])
            new_faces.append([ab, b, bc])
            new_faces.append([ca, bc, c])
            new_faces.append([ab, bc, ca])
        
        # Update faces to use indices
        verts_list = list(points_set)
        verts = np.array(verts_list)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Create index mapping
        vert_to_idx = {tuple(np.round(v, 10)): i for i, v in enumerate(verts)}
        
        faces = []
        for face in new_faces:
            idxs = []
            for v in face:
                idx = vert_to_idx[tuple(np.round(v, 10))]
                idxs.append(idx)
            faces.append(idxs)
        faces = np.array(faces)
    
    return verts

# Precompute grid points
GRID_POINTS = _generate_icosahedral_grid(4)
# Ensure grid points are unit vectors
GRID_POINTS /= np.linalg.norm(GRID_POINTS, axis=1, keepdims=True)

def _compute_covering_radius(points):
    """
    Compute the covering radius using a precomputed grid.
    """
    try:
        # Compute dot products between all grid points and all configuration points
        # dot[i, j] = dot(GRID_POINTS[i], points[j])
        dot = GRID_POINTS @ points.T  # (G, N)
        
        # For each grid point, find the maximum dot product with any configuration point
        max_dot = np.max(dot, axis=1)  # (G,)
        
        # Clip to valid range
        max_dot = np.clip(max_dot, -1.0, 1.0)
        
        # The covering radius is the maximum angular distance
        # which corresponds to the minimum max_dot
        min_max_dot = np.min(max_dot)
        min_max_dot = np.clip(min_max_dot, -1.0, 1.0)
        
        covering_radius = np.arccos(min_max_dot)
        
        return np.degrees(covering_radius)
        
    except Exception:
        return 180.0

def _compute_covering_radius_and_gradient(points):
    """
    Compute covering radius and gradient with respect to points.
    """
    try:
        # Compute dot products
        dot = GRID_POINTS @ points.T  # (G, N)
        
        # For each grid point, find the configuration point with max dot
        max_dot = np.max(dot, axis=1)  # (G,)
        max_idx = np.argmax(dot, axis=1)  # (G,)
        
        # Clip
        max_dot = np.clip(max_dot, -1.0, 1.0)
        
        # Find the grid point with minimum max_dot (worst case)
        worst_grid_idx = np.argmin(max_dot)
        worst_max_dot = max_dot[worst_grid_idx]
        worst_config_idx = max_idx[worst_grid_idx]
        
        covering_radius = np.arccos(np.clip(worst_max_dot, -1.0, 1.0))
        
        # Gradient: d(covering_radius)/d(points)
        # The gradient with respect to points[j] is:
        # -GRID_POINTS[worst_grid_idx] / sqrt(1 - dot^2) for j = worst_config_idx
        # 0 for other j
        
        grad = np.zeros_like(points)
        
        dot_val = np.clip(worst_max_dot, -1.0 + 1e-10, 1.0 - 1e-10)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        
        if denom > 1e-10:
            grad[worst_config_idx] = -GRID_POINTS[worst_grid_idx] / denom
        else:
            # If dot is close to 1, the gradient is ill-defined
            # Use a small perturbation
            grad[worst_config_idx] = -GRID_POINTS[worst_grid_idx] * 1e-6
        
        return np.degrees(covering_radius), grad
        
    except Exception:
        return 180.0, np.zeros_like(points)

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.02
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        
        # Compute covering radius and gradient
        current_radius, grad = _compute_covering_radius_and_gradient(points)
        
        # Project gradient onto tangent plane
        # For each point, the tangent plane is perpendicular to the point
        projected_grad = np.zeros_like(points)
        for i in range(N):
            dot_gp = np.dot(grad[i], points[i])
            projected_grad[i] = grad[i] - dot_gp * points[i]
        
        # Normalize gradient to have unit norm per point
        grad_norms = np.linalg.norm(projected_grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms > 1e-10, grad_norms, 1.0)
        normalized_grad = projected_grad / grad_norms
        
        # Update points: move in direction of negative gradient
        new_points = points - step_size * normalized_grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase step size slightly if making progress
            step_size = min(step_size * 1.1, 0.05)
        else:
            # If worse, revert and decrease step size
            step_size = max(step_size * 0.7, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Stagnation: try a random kick
                kick = rng.normal(0, 0.03, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
    
    return best_points
