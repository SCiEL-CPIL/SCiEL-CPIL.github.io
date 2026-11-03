import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    max_time = 145.0  # Safety margin for 150s limit
    best_points = points.copy()
    best_radius = np.inf
    
    # Pre-compute some constants
    step_size = 0.1
    damping = 0.9
    
    for _ in range(10000):
        if time.time() - start_time > max_time:
            break
            
        # Check validity
        if not np.all(np.isfinite(points)):
            break
        norms = np.linalg.norm(points, axis=1)
        if np.min(norms) < 1e-6:
            # Add noise to coincident points
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
            continue
            
        # Check if in closed hemisphere
        # A set of points is in a closed hemisphere if there exists a vector v such that v . p_i >= 0 for all i
        # This is equivalent to the origin being outside the convex hull of the points
        # If the convex hull contains the origin, they are NOT in a closed hemisphere.
        # We can check this by seeing if the convex hull has a facet with normal pointing towards the origin?
        # Actually, if the origin is inside the convex hull, the points are not in a hemisphere.
        # If the origin is outside, they are in a hemisphere.
        # So we want the origin to be INSIDE the convex hull.
        
        try:
            hull = ConvexHull(points)
        except Exception:
            # Degenerate case, add noise
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
            continue
            
        # Check if origin is inside hull
        # The origin is inside if the distance from origin to all facets is positive (assuming normals point out)
        # Hull equations: a*x + b*y + c*z + d = 0, with a*x+b*y+c*z+d <= 0 for points inside?
        # Actually, scipy's hull equations are such that the interior is where the expression is <= 0?
        # Let's check the distance of origin to planes.
        # Plane: n . x + d = 0. Distance from origin is |d| / ||n||.
        # If all d have the same sign, the origin is on one side.
        # If d values have mixed signs, the origin is inside.
        # Wait, if the origin is inside, the points are NOT in a hemisphere.
        # If the origin is outside, the points ARE in a hemisphere.
        # We want to avoid the "in a hemisphere" case.
        # So we want the origin to be INSIDE the hull.
        
        # Let's just compute the covering radius and if it's huge, it might be in a hemisphere.
        # But the prompt says "configurations inside a closed hemisphere" make the candidate invalid.
        # So we must ensure the origin is inside the convex hull.
        
        # Check if origin is inside:
        # Project origin onto each facet plane.
        # If the origin is on the "outside" of any facet (i.e., the facet normal points away from origin but origin is on the other side?), it's outside.
        # Simpler: Check if the minimum distance from origin to any facet plane is positive?
        # No, if origin is inside, it is on the "inside" of all facets.
        # The hull equations are $a_i x + b_i y + c_i z + d_i = 0$ with $a_i x + b_i y + c_i z + d_i \le 0$ for all points in hull.
        # For origin $(0,0,0)$, value is $d_i$.
        # If all $d_i \le 0$, origin is inside (or on boundary).
        # If any $d_i > 0$, origin is outside that facet, so outside hull.
        
        d_values = hull.equations[:, 3]
        if np.any(d_values > 1e-6):
            # Origin is outside, points are in a hemisphere.
            # Add noise to push them out
            points += np.random.normal(0, 0.1, points.shape)
            points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
            continue

        # Compute circumcenters and radii for each facet
        n_facets = len(hull.simplices)
        circumcenters = np.zeros((n_facets, 3))
        circumradii = np.zeros(n_facets)
        
        for k, simplex in enumerate(hull.simplices):
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            # Normal vector
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                circumcenters[k] = v1
                circumradii[k] = 0
                continue
            n = n / n_norm
            
            # Determine sign: normal should point outward.
            # Check if origin is on the negative side of the plane defined by v1, n.
            # Plane: n . x = n . v1
            # Origin: 0.
            # If n . v1 > 0, origin is on the side where n . x < n . v1, i.e., inside?
            # Actually, for a convex hull containing the origin, the facet normal points away from the origin.
            # So n . v1 > 0.
            # The circumcenter is on the sphere in the direction of n.
            c = n
            
            # Circumradius: angular distance from c to v1
            # cos(theta) = c . v1
            cos_theta = np.dot(c, v1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            circumradii[k] = np.arccos(cos_theta)
            circumcenters[k] = c

        # For each point, find the centroid of its Voronoi cell
        # Voronoi cell vertices are the circumcenters of adjacent facets
        new_points = np.zeros_like(points)
        
        # Map point index to list of facet indices
        point_to_facets = [[] for _ in range(N)]
        for k, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(k)
        
        for i in range(N):
            facets = point_to_facets[i]
            if not facets:
                new_points[i] = points[i]
                continue
                
            centers = circumcenters[facets]
            
            # Compute spherical centroid of these centers
            # Simple average is a good approximation
            centroid = np.sum(centers, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-10:
                new_points[i] = points[i]
            else:
                centroid = centroid / centroid_norm
                
            # Move point towards centroid
            # Use a step size
            new_points[i] = points[i] + step_size * (centroid - points[i])
            
        # Normalize new points
        new_points /= np.linalg.norm(new_points, axis=1)[:, np.newaxis]
        
        # Calculate covering radius
        # Max over all facets of circumradius
        current_radius = np.max(circumradii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            
        # Update points
        points = new_points
        
    return best_points
