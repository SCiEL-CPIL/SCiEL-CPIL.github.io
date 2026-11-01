"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius
    return points

def _get_worst_hole(points):
    """
    Compute the convex hull and find the facet with the largest circumradius.
    Returns:
        max_radius: The maximum circumradius in radians.
        pole: The pole (center of the worst empty cap) as a unit vector.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.array([1.0, 0.0, 0.0])
    
    max_radius = 0.0
    worst_pole = np.array([1.0, 0.0, 0.0])
    
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            
            centroid = np.mean(verts, axis=0)
            if np.dot(pole, centroid) < 0:
                pole = -pole
                
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > max_radius:
                max_radius = theta
                worst_pole = pole
    
    return max_radius, worst_pole

def construct():
    # Initialize with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Simulated Annealing Parameters
    n_iterations = 20000
    initial_temp = 0.05
    final_temp = 0.001
    decay_factor = (final_temp / initial_temp) ** (1.0 / n_iterations)
    
    current_radius, _ = _get_worst_hole(points)
    best_radius = current_radius
    best_points = points.copy()
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iterations):
        temp = initial_temp * (decay_factor ** i)
        
        # 1. Find the worst hole (pole)
        current_radius, pole = _get_worst_hole(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if len(pole) == 0 or current_radius < 1e-8:
            break

        # 2. Find the point closest to the pole
        # We want to move a point towards the pole to cover it.
        # The point closest to the pole is the one with the largest dot product with the pole.
        dots = points @ pole
        closest_idx = np.argmax(dots)
        
        # 3. Generate a new position for this point
        # Move it towards the pole.
        # Step size is proportional to temperature.
        step_size = 0.1 * temp + 0.01
        
        v = points[closest_idx]
        # Project pole onto tangent plane at v
        proj = pole - np.dot(pole, v) * v
        proj_norm = np.linalg.norm(proj)
        
        if proj_norm < 1e-12:
            # Point is very close to pole, add random noise
            noise = rng.standard_normal(3) * 0.01
            new_v = v + noise
        else:
            direction = proj / proj_norm
            # Add some randomness to avoid sticking
            noise = rng.standard_normal(3) * step_size * 0.1
            new_v = v + step_size * direction + noise
            
        new_v = new_v / np.linalg.norm(new_v)
        
        # 4. Test the move
        new_points = points.copy()
        new_points[closest_idx] = new_v
        
        new_radius, _ = _get_worst_hole(new_points)
        
        # 5. Acceptance criterion (Metropolis)
        delta = new_radius - current_radius
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            points = new_points
            current_radius = new_radius
    
    # Return the best points found
    points = best_points
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
