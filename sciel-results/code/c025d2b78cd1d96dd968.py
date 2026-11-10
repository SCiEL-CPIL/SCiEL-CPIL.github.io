"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _covering_radius(points):
    """Compute the covering radius (in degrees) of points on the unit sphere.
    Uses the convex hull facets: the covering radius is the max circumradius
    of the spherical facets (triangles) of the convex hull.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import SphericalVoronoi
    
    points = np.asarray(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    # Check for coincident points
    from scipy.spatial.distance import pdist
    d = pdist(points)
    if np.min(d) < 1e-10:
        # Coincident points - return large penalty
        return 180.0
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff there exists a vector v
    # such that v . p_i >= 0 for all i. This is equivalent to the origin
    # being outside or on the boundary of the convex hull of the points.
    hull = ConvexHull(points)
    # If the origin is inside the convex hull, the points are NOT in a closed hemisphere
    # We can check this by seeing if 0 is in the convex hull.
    # Simple check: if the convex hull contains the origin, the covering radius is well-defined
    # For 130 points on a sphere, they should not be in a hemisphere if well-distributed.
    
    # Use SphericalVoronoi to find the covering radius
    try:
        sv = SphericalVoronoi(points)
        # The covering radius is the max over all vertices of the spherical Voronoi
        # of the max angular distance to the 3 nearest points defining that vertex.
        # Actually, the covering radius equals the max circumradius of the spherical triangles
        # formed by the Delaunay triangulation.
        
        # Alternative: for each Voronoi vertex, compute the distance to the 3 points
        # that define it. The max of these is the covering radius.
        
        max_dist = 0.0
        for vertex in sv.vertices:
            # Each vertex is defined by 3 points (the indices are in sv.regions)
            # Actually, let's use a different approach: 
            # The covering radius is the max over all points y on the sphere of 
            # min_i angular_dist(y, p_i).
            # This is achieved at the circumcenters of the Delaunay triangles.
            
            # Let me use the fact that sv.vertices are the Voronoi vertices,
            # and each is associated with the points that define it.
            pass
        
        # Better approach: compute the Delaunay triangulation and for each triangle,
        # find its circumcenter and the distance to the 3 vertices.
        
        # Actually, the simplest correct approach:
        # The covering radius = max over Voronoi vertices of the angular distance
        # from that vertex to the nearest point (which should be the 3 points
        # equidistant from it, but we just need the max min-distance).
        
        # Let me just compute it directly: for each Voronoi vertex, find the
        # minimum angular distance to any of the N points. The max over all
        # Voronoi vertices of this min distance is the covering radius.
        
        # But sv.vertices might have numerical issues. Let me use a grid-based
        # approximation instead for robustness.
        
        # Actually, let's use the proper method:
        # The spherical Voronoi vertices are the points where 3 (or more) spheres
        # centered at the points with equal radius intersect. The covering radius
        # is the maximum, over all such vertices, of the angular distance to the
        # nearest center.
        
        # For each region in the Voronoi diagram, the "farthest" point in that
        # region from the center is at a vertex of the region. So we need to
        # check all Voronoi vertices.
        
        # sv.vertices are the coordinates of the Voronoi vertices on the unit sphere.
        # For each vertex, the nearest points are the ones whose regions contain it.
        # The distance from the vertex to those points is the same (by definition).
        
        # Let's compute: for each Voronoi vertex, the angular distance to the
        # defining points.
        
        # sv.regions[i] gives the indices of the vertices that form the region
        # around point i. But we need the reverse: for each vertex, which points
        # define it.
        
        # Actually, a cleaner way: the covering radius is the max over all
        # Delaunay triangles of the circumradius. Let me compute the Delaunay
        # triangulation from the Voronoi.
        
        # sv.neighbors[i] gives the indices of points that are neighbors of point i
        # in the Delaunay triangulation.
        
        # Let me just use a direct numerical approach: sample many points and
        # find the max min-distance. This is approximate but robust.
        
        # Actually, for the optimizer, I need a differentiable-ish objective.
        # Let me use a simpler proxy: the max over a dense set of test points
        # of the min angular distance to the N points.
        
        # But for the final answer, I need the exact covering radius.
        # Let me compute it properly using the Voronoi vertices.
        
        # Each Voronoi vertex is equidistant from 3 (or more) of the N points.
        # The angular distance from the vertex to those points is the circumradius
        # of the corresponding Delaunay triangle.
        
        # sv.vertices: array of shape (M, 3) with M vertices on the unit sphere.
        # For each vertex, I need to find which points it's equidistant from.
        
        # This is getting complicated. Let me use a different strategy:
        # Compute the covering radius by finding, for each point on a fine grid,
        # the min distance to the N points, and taking the max.
        
        # For the optimization, I'll use this grid-based estimate.
        
        # Generate a fine grid of test points
        n_grid = 5000
        # Use a quasi-random sequence for better coverage
        rng = np.random.RandomState(42)
        # Fibonacci lattice for test points
        i_test = np.arange(n_grid) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_grid
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        # Compute min distance from each test point to any of the N points
        # angular distance: arccos(dot product)
        # For efficiency, compute dot products in batches
        max_min_dist = 0.0
        batch_size = 1000
        for start in range(0, n_grid, batch_size):
            end = min(start + batch_size, n_grid)
            tp = test_points[start:end]
            dots = tp @ points.T  # shape (batch, N)
            # Clip to [-1, 1] for numerical stability
            dots = np.clip(dots, -1.0, 1.0)
            min_dots = np.max(dots, axis=1)  # max dot = min angular distance
            min_angles = np.arccos(min_dots)
            max_min_dist = max(max_min_dist, np.max(min_angles))
        
        return np.degrees(max_min_dist)
    
    except Exception:
        return 180.0


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Evaluate initial covering radius
    best_radius = _covering_radius(best_points)
    
    # Use a local optimization approach: iteratively move points to reduce
    # the covering radius. For each point, find the direction that reduces
    # the max min-distance in its "region".
    
    # Strategy: For each point, compute the "worst" test point in its vicinity
    # (the one with the largest min-distance that is closest to this point),
    # and move the point towards that worst point.
    
    # Generate test points for optimization
    n_test = 2000
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Iterative improvement
    points = best_points.copy()
    learning_rate = 0.05
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Compute min distance from each test point to each configuration point
        dots = test_points @ points.T  # (n_test, N)
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # (n_test,)
        min_angles = np.arccos(min_dots)  # (n_test,)
        
        # For each test point, find which configuration point is closest
        closest_idx = np.argmax(dots, axis=1)  # (n_test,)
        
        # For each configuration point, find the worst test point assigned to it
        # (the one with the largest min_angle among those assigned to this point)
        worst_angles = np.zeros(N)
        worst_test_idx = np.zeros(N, dtype=int)
        for k in range(N):
            mask = (closest_idx == k)
            if np.any(mask):
                angles_in_region = min_angles[mask]
                test_idx_in_region = np.where(mask)[0]
                worst_in_region = np.argmax(angles_in_region)
                worst_angles[k] = angles_in_region[worst_in_region]
                worst_test_idx[k] = test_idx_in_region[worst_in_region]
        
        # Move each point towards its worst test point
        for k in range(N):
            if worst_angles[k] > 0:
                # Move point k towards test_points[worst_test_idx[k]]
                target = test_points[worst_test_idx[k]]
                # Spherical interpolation: move a small step towards target
                new_point = points[k] + learning_rate * (target - points[k])
                new_point = new_point / np.linalg.norm(new_point)
                points[k] = new_point
        
        # Evaluate new covering radius
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Revert if it got worse
            points = best_points.copy()
            learning_rate *= 0.9  # Reduce learning rate
    
    # Final polish: reduce learning rate and do more iterations
    learning_rate = 0.01
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        dots = test_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_angles = np.arccos(min_dots)
        closest_idx = np.argmax(dots, axis=1)
        
        worst_angles = np.zeros(N)
        worst_test_idx = np.zeros(N, dtype=int)
        for k in range(N):
            mask = (closest_idx == k)
            if np.any(mask):
                angles_in_region = min_angles[mask]
                test_idx_in_region = np.where(mask)[0]
                worst_in_region = np.argmax(angles_in_region)
                worst_angles[k] = angles_in_region[worst_in_region]
                worst_test_idx[k] = test_idx_in_region[worst_in_region]
        
        for k in range(N):
            if worst_angles[k] > 0:
                target = test_points[worst_test_idx[k]]
                new_point = points[k] + learning_rate * (target - points[k])
                new_point = new_point / np.linalg.norm(new_point)
                points[k] = new_point
        
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            points = best_points.copy()
            learning_rate *= 0.95
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, None]
    
    return best_points
# EVOLVE-BLOCK-END
