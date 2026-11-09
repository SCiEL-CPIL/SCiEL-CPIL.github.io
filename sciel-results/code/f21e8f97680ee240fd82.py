import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _circumradius_3d(p1, p2, p3):
    """
    Compute the circumradius of a spherical triangle defined by 3 unit vectors.
    """
    # Side lengths
    # We need the circumcenter.
    # Vector direction of circumcenter:
    v = np.cross(p3 - p2, p2 - p1)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return 0.0
    c_center = v / norm
    cos_r = np.clip(np.dot(c_center, p1), -1.0, 1.0)
    return np.arccos(cos_r)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin
    
    # Initialization: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_radius = float('inf')
    best_points = points.copy()
    
    alpha = 0.05 # Step size
    stagnation_count = 0
    last_improvement = best_radius
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
            
            # Compute circumradii
            radii = np.array([_circumradius_3d(points[s[0]], points[s[1]], points[s[2]]) for s in simplices])
            current_radius = np.max(radii)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                last_improvement = best_radius
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # If stagnating, perturb
            if stagnation_count > 200:
                points += np.random.normal(0, 0.05, points.shape)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                stagnation_count = 0
                continue
            
            # Lloyd's Step
            new_points = points.copy()
            for i in range(N):
                neighbors = []
                for s in simplices:
                    if i in s:
                        for j in s:
                            if j != i:
                                neighbors.append(j)
                
                if neighbors:
                    avg_neighbor = np.mean(points[neighbors], axis=0)
                    new_points[i] = points[i] + alpha * (avg_neighbor - points[i])
                    norm = np.linalg.norm(new_points[i])
                    if norm > 1e-10:
                        new_points[i] /= norm
            
            points = new_points
            
        except Exception:
            points += np.random.normal(0, 0.05, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

    return best_points
