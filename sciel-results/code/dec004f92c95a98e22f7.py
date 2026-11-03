"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return np.pi  # invalid
    
    # Check if all points are in a closed hemisphere
    # This can be checked by seeing if there's a plane through origin separating them
    # A simpler check: if the convex hull doesn't contain the origin
    try:
        hull = ConvexHull(points)
        # Check if origin is inside the convex hull
        # The origin is inside if for all facets, the origin is on the same side
        # or we can use the fact that if points are in a hemisphere, 
        # there exists a vector v such that v·p_i >= 0 for all i
        # This is equivalent to the origin not being in the interior of the convex hull
        # Use scipy to check
        from scipy.spatial import Delaunay
        # Actually, let's use a different approach
        # If all points are in a hemisphere, the convex hull won't contain the origin
        # We can check this by solving: is 0 in conv(points)?
        # This is a linear programming problem
        from scipy.optimize import linprog
        # Minimize 0 subject to sum(lambda_i * p_i) = 0, sum(lambda_i) = 1, lambda_i >= 0
        # If feasible, origin is in convex hull
        A_eq = np.vstack([points.T, np.ones(N)])
        b_eq = np.zeros(3 + 1)
        res = linprog(np.zeros(N), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*N, method='highs')
        if res.status == 0:
            # Origin is in convex hull, so points are NOT all in one hemisphere
            pass
        else:
            # Origin not in convex hull, points might be in a hemisphere
            # But this check is not sufficient; let's use a more robust method
            # Actually, if the origin is NOT in the convex hull, the points ARE in some hemisphere
            # Wait, that's the opposite. If origin is in convex hull, points are NOT in any hemisphere.
            # If origin is NOT in convex hull, points ARE in some hemisphere.
            # For a valid covering, we need points NOT all in one hemisphere, so origin must be in convex hull.
            return np.pi  # invalid: points in a hemisphere
    except Exception:
        pass
    
    # Compute the covering radius using the convex hull facets
    # The covering radius is the maximum over all facets of the angular distance
    # from the facet's circumcenter to the farthest point on the facet
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        
        for simplex in hull.simplices:
            facet_points = points[simplex]
            # Compute the circumcenter of the facet (triangle for 3D)
            # For a triangle with vertices a, b, c on the unit sphere,
            # the circumcenter is in the direction of the normal, at distance 1/cos(theta)
            # where theta is the angle between the normal and any vertex
            # Actually, the circumcenter of a spherical triangle is the point on the sphere
            # equidistant from the three vertices. It's the normalized cross product of...
            
            # For a triangle a, b, c, the circumcenter can be found as:
            # It lies on the plane perpendicular to the face normal passing through the centroid? No.
            # The circumcenter of a spherical triangle is the point p such that |p-a|=|p-b|=|p-c|
            # This means p is in the direction of the normal to the plane of a,b,c
            # scaled appropriately.
            
            # Let n = (b-a) x (c-a), normalized
            # The circumcenter is n / (n·a) if n·a > 0, else -n / (-n·a)
            # Wait, let's think more carefully.
            
            # For a triangle on the unit sphere, the circumcenter is the point on the sphere
            # that is equidistant from all three vertices. This point lies on the line through
            # the origin and the circumcenter of the planar triangle formed by a, b, c.
            
            # The circumcenter of the planar triangle (a,b,c) can be computed, then normalized.
            
            a, b, c = facet_points
            # Compute circumcenter of triangle a, b, c in 3D
            # Using the formula: circumcenter = (|a|^2(b×c) + |b|^2(c×a) + |c|^2(a×b)) / (2(a·(b×c)))
            # Since |a|=|b|=|c|=1, this simplifies to (b×c + c×a + a×b) / (2(a·(b×c)))
            
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            ab = np.cross(a, b)
            denom = 2.0 * np.dot(a, np.cross(b, c))
            
            if abs(denom) < 1e-12:
                continue  # degenerate triangle
            
            circumcenter = (bc + ca + ab) / denom
            cc_norm = np.linalg.norm(circumcenter)
            if cc_norm < 1e-12:
                continue
            circumcenter = circumcenter / cc_norm
            
            # The angular distance from circumcenter to any vertex is the circumradius
            # of this facet
            dist = np.arccos(np.clip(np.dot(circumcenter, a), -1.0, 1.0))
            max_radius = max(max_radius, dist)
        
        return max_radius
    except Exception:
        # Fallback: use a grid-based approximation
        # This is a rough estimate
        lat = np.linspace(-np.pi/2, np.pi/2, 100)
        lon = np.linspace(0, 2*np.pi, 200)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        y = np.column_stack([
            np.cos(lat_grid) * np.cos(lon_grid).ravel(),
            np.cos(lat_grid) * np.sin(lon_grid).ravel(),
            np.sin(lat_grid).ravel()
        ])
        
        # Compute min distance from each y to any point
        # Using broadcasting
        diffs = y[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        max_min_dist = np.max(min_dists)
        # Convert chordal distance to angular distance
        # chord = 2*sin(theta/2), so theta = 2*arcsin(chord/2)
        return 2.0 * np.arcsin(np.clip(max_min_dist / 2.0, 0, 1))


def _lloyd_relaxation(points, n_iters=100, time_limit=10.0):
    """Perform Lloyd's relaxation to improve the covering."""
    points = np.array(points, dtype=float)
    start_time = time.time()
    
    for _ in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # For each point, find its Voronoi cell (points closer to it than to any other)
        # and move it to the centroid of its cell
        new_points = np.zeros_like(points)
        
        for i in range(N):
            # Find points on the sphere closer to points[i] than to any other
            # This is the Voronoi cell of points[i]
            # Approximate by sampling
            # Use a local grid around points[i]
            
            # Generate random points on the sphere
            n_samples = 2000
            theta = np.random.uniform(0, 2*np.pi, n_samples)
            phi = np.arccos(np.random.uniform(-1, 1, n_samples))
            samples = np.column_stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            # Compute distances to all points
            dists = np.zeros((n_samples, N))
            for j in range(N):
                dists[:, j] = np.linalg.norm(samples - points[j], axis=1)
            
            # Find samples closest to point i
            mask = np.argmin(dists, axis=1) == i
            if np.sum(mask) > 0:
                # Move to centroid of the Voronoi cell
                new_points[i] = np.mean(samples[mask], axis=0)
            else:
                new_points[i] = points[i]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        # Check for improvement
        old_radius = _covering_radius(points)
        new_radius = _covering_radius(new_points)
        
        if new_radius < old_radius:
            points = new_points
        else:
            break
    
    return points


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple random starts and relaxations
    n_starts = 5
    
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if s == 0:
            current = points.copy()
        else:
            # Random start
            current = np.random.randn(N, 3)
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Lloyd relaxation
        current = _lloyd_relaxation(current, n_iters=50, time_limit=15.0)
        
        # Evaluate
        radius = _covering_radius(current)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current.copy()
    
    # Final check
    if best_points is None:
        best_points = points
    
    return best_points
# EVOLVE-BLOCK-END
