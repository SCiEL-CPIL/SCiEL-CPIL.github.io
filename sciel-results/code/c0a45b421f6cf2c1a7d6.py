"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the maximum circumradius of the facets
    of the convex hull of the points (the empty cap radius).
    
    We compute this via the convex hull: for each triangular facet, find the 
    circumradius of the triangle on the unit sphere.
    """
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return np.pi  # invalid
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff there exists a vector v
    # such that v . p_i >= 0 for all i. Equivalently, the origin is not in
    # the interior of the convex hull.
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    # The covering radius is the max over all hull facets of the angular distance
    # from the facet's circumcenter (projected to sphere) to the farthest vertex.
    # For a triangle on the unit sphere with vertices a, b, c, the circumradius
    # (angular) is half the maximum angular distance between any two vertices
    # only if the triangle is right... no, we need the actual spherical circumradius.
    
    # For a triangle on the unit sphere, the circumradius R (angular) satisfies:
    # The circumcenter is the point on the sphere equidistant (angularly) from a, b, c.
    # The angular circumradius is the angle at the circumcenter to each vertex.
    
    # A simpler approach: for each facet (triangle), compute the maximum distance
    # from the triangle's interior to the vertices. The farthest point in the 
    # spherical triangle from the "center" is one of the vertices, and the 
    # circumradius is the maximum angular distance from the circumcenter to a vertex.
    
    # Actually, the covering radius for a facet is the radius of the smallest 
    # spherical cap centered at a point on the sphere that covers the entire 
    # facet. For a spherical triangle, this is the maximum of:
    # - The angular circumradius (if circumcenter is inside the triangle)
    # - Half the longest side (if the triangle is obtuse, the circumcenter is outside)
    
    # Let's compute it properly. For a triangle with side lengths (angular) a, b, c:
    # The circumradius R satisfies: sin(R) = (a*b*c) / (4 * Area) where Area is 
    # the spherical excess... this is getting complicated.
    
    # Alternative: use the fact that for a facet of the convex hull, the covering
    # radius is the maximum distance from the facet's plane intersection with the 
    # sphere to the vertices. 
    
    # Simplest correct approach: for each triangular facet, compute the spherical 
    # circumradius. For a triangle on the unit sphere with vertices v1, v2, v3:
    # The circumcenter c satisfies |c-v1|_angular = |c-v2|_angular = |c-v3|_angular = R.
    # In 3D, the circumcenter of the triangle (as a planar triangle) projected to
    # the sphere gives the spherical circumcenter if it's inside.
    
    # Let's use a numerical approach: for each facet, find the point on the sphere
    # that maximizes the minimum distance to the three vertices. This is the 
    # spherical circumcenter.
    
    # Actually, the standard result: for a convex polyhedron inscribed in the unit 
    # sphere, the covering radius is the maximum over all facets of the angular 
    # distance from the facet's circumcenter to its vertices. For a triangular 
    # facet with vertices a, b, c on the unit sphere, the circumradius is:
    
    # Let d_ab = angular distance between a and b, etc.
    # The circumradius R of a spherical triangle with sides a, b, c satisfies:
    # cos(2R) = (cos(a)cos(b)cos(c) - 2cos(a)cos(b)cos(c) + 1) / ... 
    
    # This is complex. Let me use a different approach:
    # The covering radius is the maximum over all points y on the sphere of 
    # min_i angular_distance(y, p_i). 
    # For the convex hull approach: for each facet (triangle) of the hull, the
    # worst point within that facet's Voronoi cell is the circumcenter of the 
    # triangle (if it lies within the triangle) or the midpoint of the longest 
    # edge (if the triangle is obtuse).
    
    # For a planar triangle with vertices a, b, c, the circumradius is 
    # abc/(4*Area). On the sphere, we can compute the circumcenter as follows:
    # The circumcenter of the spherical triangle is in the direction of 
    # (a x b) x (b x c) + ... 
    
    # Let me just compute it numerically for each facet.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        # tri is a 3x3 array (3 vertices, 3D)
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute angular distances
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # For the circumradius: if the triangle is "acute" (all angles < pi/2),
        # the circumcenter is inside and R is the circumradius.
        # If obtuse, the farthest point from the circumcenter (which is outside)
        # is the midpoint of the longest side, so R = longest_side / 2.
        
        # Check if obtuse: a triangle on the sphere is obtuse if one side > pi/2
        # (since the sum of angles > pi, but side > pi/2 implies the opposite 
        # angle > pi/2 in some sense). Actually, for a spherical triangle, if
        # any side >= pi/2, the circumcenter may be outside.
        
        # Simpler: compute the circumcenter direction and check if it's inside.
        # The circumcenter of a spherical triangle is in the direction:
        # c_dir = (a x b) x (b x c) + (b x c) x (c x a) + (c x a) x (a x b)
        # ... this is getting complicated. Let me use a robust numerical method.
        
        # The circumradius R satisfies: for the circumcenter u (unit vector),
        # u . a = u . b = u . c = cos(R).
        # So u is equidistant from a, b, c in dot product.
        # (u . a)^2 = (u . b)^2 => u . (a - b) = 0 or u . (a + b) = 0
        # Actually: u . a = u . b => u . (a - b) = 0
        # u . b = u . c => u . (b - c) = 0
        # So u is perpendicular to (a-b) and (b-c), i.e., u is parallel to (a-b) x (b-c).
        
        u_dir = np.cross(a - b, b - c)
        u_norm = np.linalg.norm(u_dir)
        if u_norm < 1e-12:
            # Degenerate triangle
            R = max(d_ab, d_bc, d_ca) / 2
        else:
            u = u_dir / u_norm
            # Check both directions
            cos_R1 = np.clip(np.dot(u, a), -1, 1)
            cos_R2 = np.clip(-np.dot(u, a), -1, 1)
            R1 = np.arccos(cos_R1)
            R2 = np.arccos(cos_R2)
            R = min(R1, R2)  # The circumcenter is the closer one
            
            # But we need to check if this circumcenter is actually inside the 
            # spherical triangle. If not, the covering radius for this facet is
            # the maximum distance to the nearest vertex, which would be half the
            # longest side.
            
            # Check if u is inside the triangle: u should be a positive combination
            # of a, b, c (for the planar triangle) or equivalently, the barycentric
            # coordinates should be positive.
            
            # For a point u on the sphere, it's inside the spherical triangle abc
            # if u is on the same side of each edge as the opposite vertex.
            # Edge ab: normal is a x b. c should be on the same side as u.
            n_ab = np.cross(a, b)
            n_bc = np.cross(b, c)
            n_ca = np.cross(c, a)
            
            # Ensure consistent orientation
            # The facet normal should point outward
            n_facet = np.cross(b - a, c - a)
            if np.dot(n_facet, a) < 0:
                n_facet = -n_facet
                n_ab = -n_ab
                n_bc = -n_bc
                n_ca = -n_ca
            
            # u is inside if:
            # dot(n_ab, u) and dot(n_ab, c) have the same sign
            # dot(n_bc, u) and dot(n_bc, a) have the same sign
            # dot(n_ca, u) and dot(n_ca, b) have the same sign
            
            inside = True
            if np.dot(n_ab, u) * np.dot(n_ab, c) < 0:
                inside = False
            if np.dot(n_bc, u) * np.dot(n_bc, a) < 0:
                inside = False
            if np.dot(n_ca, u) * np.dot(n_ca, b) < 0:
                inside = False
            
            if not inside:
                # Circumcenter is outside; the covering radius is half the longest side
                R = max(d_ab, d_bc, d_ca) / 2
        
        if R > max_radius:
            max_radius = R
    
    return max_radius


def _lloyd_relaxation(points, max_iters=200, tol=1e-10):
    """Lloyd's algorithm for spherical k-means (Voronoi relaxation).
    
    For each point, find the centroid of its Voronoi cell (spherical mean) 
    and move the point there.
    """
    points = np.array(points, dtype=np.float64)
    
    for iteration in range(max_iters):
        # Compute nearest neighbor for each point to determine Voronoi cells
        # For Lloyd's on the sphere, we need the spherical centroid of each cell.
        # Approximate by using a fine grid.
        
        # Generate a fine spherical grid
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian
        grid_x = np.sin(THETA) * np.cos(PHI)
        grid_y = np.sin(THETA) * np.sin(PHI)
        grid_z = np.cos(THETA)
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        
        # Normalize
        grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
        grid_norms[grid_norms < 1e-12] = 1.0
        grid = grid / grid_norms
        
        # Find nearest point for each grid point
        dists = cdist(grid, points)  # Euclidean distance
        nearest = np.argmin(dists, axis=1)
        
        # Compute new positions as spherical means of Voronoi cells
        new_points = np.zeros_like(points)
        for i in range(N):
            mask = nearest == i
            if np.any(mask):
                # Spherical mean: normalize the vector sum
                cell_points = grid[mask]
                mean_vec = np.sum(cell_points, axis=0)
                mean_norm = np.linalg.norm(mean_vec)
                if mean_norm > 1e-12:
                    new_points[i] = mean_vec / mean_norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    
    return points


def _generate_initial_configurations(rng):
    """Generate multiple initial configurations."""
    configs = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Random configurations
    for _ in range(10):
        pts = rng.standard_normal((N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    # Perturbed Fibonacci
    for _ in range(5):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        noise = rng.standard_normal((N, 3)) * 0.1
        pts += noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    return configs


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate initial configurations
    configs = _generate_initial_configurations(rng)
    
    best_points = None
    best_radius = np.pi
    
    for config in configs:
        if time.time() - start_time > DEADLINE:
            break
        
        # Lloyd relaxation
        relaxed = _lloyd_relaxation(config, max_iters=100)
        
        # Compute covering radius
        radius = _covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
    
    # Try to improve with local optimization on the best configuration
    if best_points is not None and time.time() - start_time < DEADLINE - 30:
        # Use scipy minimize with a soft-max surrogate
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            # Check for coincident points
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                        return np.pi
            
            # Use a grid-based soft-max approximation
            n_theta = 100
            n_phi = 200
            theta = np.linspace(0, np.pi, n_theta)
            phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
            THETA, PHI = np.meshgrid(theta, phi)
            
            grid_x = np.sin(THETA) * np.cos(PHI)
            grid_y = np.sin(THETA) * np.sin(PHI)
            grid_z = np.cos(THETA)
            grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
            grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
            grid_norms[grid_norms < 1e-12] = 1.0
            grid = grid / grid_norms
            
            # Angular distances
            dots = grid @ pts.T  # (n_grid, N)
            dots = np.clip(dots, -1, 1)
            angles = np.arccos(dots)  # (n_grid, N)
            
            # Min distance from each grid point to nearest config point
            min_angles = np.min(angles, axis=1)  # (n_grid,)
            
            # Soft-max approximation of max
            beta = 50.0
            log_sum_exp = np.log(np.sum(np.exp(-beta * min_angles), axis=0)) / beta
            soft_max = -log_sum_exp
            
            return soft_max
        
        x0 = best_points.ravel()
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < best_radius:
                opt_pts = result.x.reshape(N, 3)
                opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_radius = _covering_radius(opt_pts)
                if opt_radius < best_radius:
                    best_radius = opt_radius
                    best_points = opt_pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
