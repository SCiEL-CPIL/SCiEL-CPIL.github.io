import numpy as np
import time
from scipy.spatial import Delaunay

N = 130

def _circumradius_sphere(p1, p2, p3):
    # p1, p2, p3 are unit vectors
    # The circumcenter of a spherical triangle is the normalized sum of the points
    # if the triangle is small enough (less than 180 degrees).
    # Actually, the circumcenter is the intersection of the perpendicular bisectors.
    # For a spherical triangle, the circumcenter is the normalized vector:
    # C = (p1 * sin(A) + p2 * sin(B) + p3 * sin(C)) / sin(A+...)? No.
    # Simpler: The circumcenter is the point on the sphere equidistant from p1, p2, p3.
    # It is the normalized sum of the unit vectors if the origin is inside the triangle?
    # No, that's the centroid.
    # The circumcenter is the solution to x . p1 = x . p2 = x . p3.
    # x is in the plane defined by p1, p2, p3? No, x is on the sphere.
    # The plane of the triangle is normal to n = p1 x p2 x p3 ? No.
    # The circumcenter lies on the normal to the plane of the triangle passing through the origin?
    # No, the circumcenter is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle?
    # Yes, if the triangle is small.
    # Let's compute the circumcenter of the planar triangle p1, p2, p3.
    # Then normalize it.
    
    a = p1 - p2
    b = p1 - p3
    # Circumcenter of triangle with vertices p1, p2, p3 in 3D (projected to 2D plane)
    # We can use the formula for circumcenter in 3D for a triangle in a plane.
    # Or just use the fact that C is orthogonal to (p1-p2) and (p1-p3) in the plane?
    # C . (p1-p2) = 0 and C . (p1-p3) = 0? No, C is on the sphere.
    # C . p1 = C . p2 = C . p3 = r.
    # C . (p1-p2) = 0 and C . (p1-p3) = 0.
    # So C is orthogonal to the plane of the triangle? No, C is in the plane of the triangle?
    # No, the circumcenter of a spherical triangle is the normalized sum of the points?
    # Let's check. If p1, p2, p3 are orthogonal, C = (1,1,1)/sqrt(3).
    # d(C, p1) = arccos(1/sqrt(3)) ~ 54.7 deg.
    # d(p1, p2) = 90 deg.
    # This seems correct.
    
    # General formula:
    # C = (p1 * |p2 x p3| + p2 * |p3 x p1| + p3 * |p1 x p2|) / |...|
    # This is the normalized sum of the points weighted by the areas of the opposite sub-triangles?
    # Actually, the circumcenter is the normalized vector:
    # C = (p1 * sin(d23) + p2 * sin(d31) + p3 * sin(d12))
    # where d23 is the angular distance between p2 and p3.
    
    d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
    d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
    d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    
    w1 = np.sin(d23)
    w2 = np.sin(d31)
    w3 = np.sin(d12)
    
    C = w1 * p1 + w2 * p2 + w3 * p3
    norm = np.linalg.norm(C)
    if norm < 1e-10:
        return 0.0, np.zeros(3)
    C = C / norm
    # The circumradius is the distance from C to any of the points, e.g., p1
    r = np.arccos(np.clip(np.dot(C, p1), -1.0, 1.0))
    return r, C

def _covering_radius(points):
    # points: (N, 3)
    tri = Delaunay(points)
    max_r = 0.0
    max_idx = -1
    for simplex in tri.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        r, C = _circumradius_sphere(p1, p2, p3)
        if r > max_r:
            max_r = r
            max_idx = simplex
    return max_r, max_idx

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_points = points.copy()
    best_radius = _covering_radius(points)[0]
    
    # Local search
    alpha = 0.01 # Step size
    
    while time.time() < deadline:
        radius, max_simplex = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if max_simplex is None or max_simplex == -1:
            break
            
        p1 = points[max_simplex[0]]
        p2 = points[max_simplex[1]]
        p3 = points[max_simplex[2]]
        
        _, C = _circumradius_sphere(p1, p2, p3)
        
        # Move points towards C
        # This reduces the circumradius of this triangle
        # But might increase others
        # We use a small step
        step = alpha
        points[max_simplex[0]] = p1 + step * (C - p1)
        points[max_simplex[1]] = p2 + step * (C - p2)
        points[max_simplex[2]] = p3 + step * (C - p3)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Occasionally, add a small random perturbation to escape local minima
        if np.random.random() < 0.05:
            idx = np.random.randint(N)
            pert = np.random.normal(0, 0.01, 3)
            points[idx] += pert
            points[idx] /= np.linalg.norm(points[idx])
            
    return best_points
