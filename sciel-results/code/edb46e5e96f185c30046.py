import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(np.dot(normal, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def construct():
    # Approach: Simulated Annealing minimizing covering radius directly
    
    def tangent_direction(point):
        # Generate a random vector in the tangent plane at 'point'
        v = np.random.randn(3)
        v = v - np.dot(v, point) * point
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return v / norm

    best_points = None
    best_radius = np.pi

    # Run multiple restarts to avoid local minima
    for restart in range(10):
        # Initialize with Fibonacci spiral + noise
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise to break symmetry
        points += 0.1 * np.random.randn(N, 3)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
        
        current_radius = _covering_radius(points)
        
        # Simulated Annealing Parameters
        T = 0.05
        steps = 2000
        
        for step in range(steps):
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Adaptive move size: smaller as T decreases
            move_size = 0.05 * (T / 0.05) + 0.01
            dir_vec = tangent_direction(points[idx])
            new_point = points[idx] + move_size * dir_vec
            new_point /= np.linalg.norm(new_point)
            
            # Store old point
            old_point = points[idx].copy()
            points[idx] = new_point
            
            new_radius = _covering_radius(points)
            
            # Acceptance criteria
            if new_radius < current_radius or np.random.rand() < np.exp(-(new_radius - current_radius) / T):
                current_radius = new_radius
            else:
                # Reject move
                points[idx] = old_point
            
            # Cool down
            T *= 0.999
            
            # If we found a new global best, update
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        # If current is better than best, update
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

    return best_points
