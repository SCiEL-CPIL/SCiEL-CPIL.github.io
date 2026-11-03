import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

def _gram_matrix_gradient(points):
    """
    Compute the gradient of the objective function based on the Gram matrix.
    Objective: minimize trace(G^2) where G = points @ points.T
    This encourages uniform distribution of points on the sphere.
    
    The gradient with respect to point i is:
    d/dp_i [trace(G^2)] = 2 * (G^2 @ p_i) projected onto tangent plane
    """
    G = points @ points.T
    G2 = G @ G
    grad = 2.0 * (G2 @ points)
    # Project gradient onto tangent plane: grad_i - (grad_i . p_i) * p_i
    dot_gp = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_gp * points
    return grad_tangent

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
    step_size = 0.01
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 100
    eval_interval = 50  # Evaluate covering radius every 50 iterations
    
    iteration = 0
    
    while time.time() < deadline:
        # Compute gradient using Gram matrix (fast, no ConvexHull)
        grad = _gram_matrix_gradient(points)
        
        # Update points
        new_points = points - step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for coincident points or numerical issues
        if not np.all(np.isfinite(new_points)):
            step_size = max(step_size * 0.5, min_step_size)
            continue
        
        # Check if points are too close (coincident)
        G_check = new_points @ new_points.T
        np.fill_diagonal(G_check, 0)
        if np.min(G_check) > 1 - 1e-6:
            # Points too close, add random kick
            kick = rng.normal(0, 0.01, new_points.shape)
            new_points += kick
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        points = new_points
        iteration += 1
        
        # Periodically evaluate the actual covering radius
        if iteration % eval_interval == 0:
            current_radius = _compute_covering_radius(points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                no_improve_count = 0
                # Slightly increase step size if making progress
                step_size = min(step_size * 1.1, 0.05)
            else:
                no_improve_count += 1
                # Decrease step size if not improving
                step_size = max(step_size * 0.9, min_step_size)
                
                if no_improve_count > max_no_improve:
                    # Stagnation: try a random kick
                    kick = rng.normal(0, 0.03, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = 0.005
                    no_improve_count = 0
                    current_radius = _compute_covering_radius(points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            # Scale step size to keep updates reasonable
            step_size = min(step_size, 0.01 / (grad_norm + 1e-8))
    
    # Final evaluation
    final_radius = _compute_covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_points
