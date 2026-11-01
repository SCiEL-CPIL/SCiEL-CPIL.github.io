import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi # Invalid, treat as large radius
    points = points / norms[:, np.newaxis]
    
    # Check if points are in a hemisphere.
    # If they are, the origin is not in the interior of the convex hull,
    # and the "covering radius" concept via hull facets is tricky (it would be pi/2 or more effectively).
    # The task says configurations inside a closed hemisphere are invalid.
    # We can check if there exists a vector v such that v . p_i > 0 for all i.
    # This is equivalent to checking if the origin is inside the convex hull.
    # If the origin is NOT inside, the covering radius is at least pi/2.
    # We can use a linear programming check or just check if the convex hull contains the origin.
    # A simpler heuristic: if the max distance from origin to a facet is negative?
    # Actually, if the origin is outside, the "facets" visible from the origin don't cover the sphere.
    # We will return a large penalty if the origin is not strictly inside.
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    # Check if origin is inside the hull.
    # The hull contains the origin if the minimum distance from the origin to any facet is > 0.
    # Wait, the distance from origin to facet plane is d. If d > 0 for all facets (with normals pointing out),
    # then the origin is inside.
    # ConvexHull normals point outward.
    # The plane equation is n . x = d.
    # If the origin (0,0,0) is inside, then n . 0 < d is not the condition.
    # The condition is that the origin is on the same side of all facets as the interior.
    # For a convex hull of points on the sphere, if the origin is inside,
    # then for every facet with outward normal n and plane n . x = d, we must have d > 0.
    # Because the origin satisfies n . 0 = 0 < d.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Compute normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-9:
            continue
        normal = normal / norm_n
        
        # Distance from origin to the plane
        d = np.dot(normal, p1)
        
        # If d <= 0, the origin is not inside the hull (or on the boundary).
        # This implies the points are in a hemisphere.
        if d <= 1e-9:
            return np.pi # Penalty
        
        # Circumradius R: cos(R) = d / |normal|? No.
        # normal is unit vector.
        # The circumcenter c is in the direction of normal.
        # c = normal.
        # Distance from c to p1 is arccos(c . p1).
        cos_R = np.dot(normal, p1)
        # Due to numerical errors, cos_R might be > 1 or < -1
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        if R > max_radius:
            max_radius = R
            
    return max_radius

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    def objective(x_flat):
        points = x_flat.reshape(N, 3)
        # Normalize to unit sphere to ensure constraint satisfaction during optimization
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        points = points / norms[:, np.newaxis]
        return _covering_radius(points)
    
    # Use Nelder-Mead for non-smooth objective
    # maxiter needs to be tuned. 1000 iterations might be too slow if each eval is slow.
    # ConvexHull for N=50 is fast (~ms). 1000 iters * 50 points * ms = manageable.
    # Let's try 200 iterations first to stay safe within 120s.
    try:
        res = minimize(objective, x0_flat, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.success or res.fun < objective(x0_flat):
            x_opt = res.x
        else:
            x_opt = x0_flat
    except Exception:
        x_opt = x0_flat
        
    # Reshape and normalize
    points = x_opt.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points
