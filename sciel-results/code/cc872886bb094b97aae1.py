"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import math

def construct():
    # Equilateral triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, math.sqrt(3)/2])
    
    # Function to check if a point is inside the triangle (with small margin)
    def is_inside_triangle(p, margin=0.02):
        # Barycentric coordinates
        # Using area method or cross products
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, A, B)
        d2 = sign(p, B, C)
        d3 = sign(p, C, A)
        
        has_neg = (d1 < -margin) or (d2 < -margin) or (d3 < -margin)
        has_pos = (d1 > margin) or (d2 > margin) or (d3 > margin)
        
        # For robustness with margin, we just check if all are >= -margin
        return d1 >= -margin and d2 >= -margin and d3 >= -margin
    
    def project_to_triangle(p, margin=0.02):
        # Simple projection: if outside, move towards centroid
        centroid = (A + B + C) / 3
        if is_inside_triangle(p, margin=0.0):
            return p
        # Move towards centroid
        for t in np.linspace(0.0, 1.0, 50):
            q = p + t * (centroid - p)
            if is_inside_triangle(q, margin=margin):
                return q
        return centroid
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 points."""
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    area = 0.5 * cross
                    if area < min_area:
                        min_area = area
        return min_area
    
    def objective(points_flat):
        """Objective function to minimize negative min area (i.e., maximize min area)."""
        points = points_flat.reshape(-1, 2)
        # Check if all points are inside triangle
        for p in points:
            if not is_inside_triangle(p, margin=0.0):
                return 1e10  # Large penalty
        area = min_triangle_area(points)
        return -area
    
    # Generate initial points using structured grid + jitter
    def generate_initial_points(seed):
        rng = np.random.RandomState(seed)
        points = []
        
        # Create a grid of points inside the triangle
        # Use barycentric coordinates
        n_grid = 5
        candidates = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                for k in range(n_grid - i - j):
                    if i + j + k == n_grid - 1:
                        # Barycentric coordinates
                        a = i / (n_grid - 1)
                        b = j / (n_grid - 1)
                        c = k / (n_grid - 1)
                        p = a * A + b * B + c * C
                        candidates.append(p)
        
        # Shuffle and pick N points with some jitter
        rng.shuffle(candidates)
        for p in candidates:
            if len(points) >= N:
                break
            # Add small jitter
            jitter = rng.uniform(-0.05, 0.05, 2)
            p_jittered = p + jitter
            p_jittered = project_to_triangle(p_jittered, margin=0.02)
            points.append(p_jittered)
        
        # If we don't have enough, add random points
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = u * A + v * B + (1 - u - v) * C
                p = project_to_triangle(p, margin=0.02)
                points.append(p)
        
        return np.array(points[:N])
    
    # Multi-start Nelder-Mead optimization
    best_points = None
    best_area = -1e10
    
    n_starts = 15
    for start in range(n_starts):
        init_points = generate_initial_points(start * 100 + 42)
        init_flat = init_points.flatten()
        
        # Nelder-Mead optimization
        result = minimize(
            objective,
            init_flat,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        opt_points = result.x.reshape(-1, 2)
        
        # Check validity
        valid = True
        for p in opt_points:
            if not is_inside_triangle(p, margin=0.0):
                valid = False
                break
        
        if valid:
            area = min_triangle_area(opt_points)
            if area > best_area:
                best_area = area
                best_points = opt_points
    
    if best_points is None:
        # Fallback to initial points
        best_points = generate_initial_points(42)
    
    return best_points
# EVOLVE-BLOCK-END
