import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _time_ok(start, deadline):
    return time.time() < deadline

def _compute_covering_radius(points):
    """
    Computes the covering radius by finding the Voronoi vertices (Delaunay circumcenters)
    and taking the maximum angular distance from the origin to these vertices (projected to sphere).
    Actually, the covering radius is the max distance from any point on the sphere to the nearest site.
    For a spherical Voronoi diagram, the worst point is a Voronoi vertex.
    The distance from the origin to the Voronoi vertex (in 3D Euclidean space) is related to the angular radius.
    If v is the Voronoi vertex (unit vector is not necessarily the vertex, the vertex is in R3),
    the angular distance theta satisfies cos(theta) = 1 / ||v||.
    So theta = arccos(1 / ||v||).
    """
    try:
        hull = ConvexHull(points)
        # hull.equations are the planes of the facets.
        # For a convex polytope containing the origin, the normal to a facet at distance d from origin
        # corresponds to the Voronoi vertex.
        # The equation is n.x + d = 0. Since origin is inside, d < 0 if n points out.
        # Let's use hull.equations which are [a, b, c, d] such that a*x+b*y+c*z+d <= 0 for all points.
        # The vertex of the Voronoi diagram is the point closest to the origin on the plane?
        # No, the Voronoi vertex is the intersection of the perpendiculars from the sites.
        # For the convex hull of points on the sphere, the facets correspond to the Voronoi cells.
        # The circumcenter of a facet (triangle of 3 points on sphere) is the Voronoi vertex.
        # We can compute the circumcenter for each facet.
        
        vertices = hull.vertices
        simplices = hull.simplices
        
        max_radius_deg = 0.0
        max_vertex = None
        
        for simplex in simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Circumcenter of triangle p1, p2, p3
            # Solve for x such that |x-p1|^2 = |x-p2|^2 = |x-p3|^2
            # This is equivalent to finding the point equidistant from p1, p2, p3.
            # Since they are on the unit sphere, the circumcenter lies on the line passing through origin and the normal to the plane of the triangle.
            # Normal n = (p2-p1) x (p3-p1)
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # The circumcenter C is at distance d from origin along n.
            # |C - p1|^2 = |C|^2 - 2 C.dot(p1) + |p1|^2
            # Let C = k * n.
            # k^2 - 2 k n.dot(p1) + 1 = 0
            # We want the root closer to the sphere (smaller k? No, the circumcenter of a spherical triangle is on the normal).
            # Actually, for a covering problem, the Voronoi vertex is the point on the normal line that is equidistant to p1, p2, p3.
            # There are two such points. The one relevant for the Voronoi cell containing the origin?
            # The Voronoi cell is the region closer to p1 than others.
            # The vertex is the intersection of bisecting planes.
            # For points on the sphere, the circumcenter of the triangle formed by the points on the sphere 
            # projects to the Voronoi vertex.
            
            # Let's just compute the distance from origin to the plane of the triangle.
            # Distance d_plane = n.dot(p1)
            # The circumcenter C satisfies C = (1 / (2 * d_plane)) * (p1 + p2 + p3) ? No.
            # Let's use the formula for circumcenter of 3 points on unit sphere.
            # C = ( (p2 x p3) * |p1-p3|^2 + (p3 x p1) * |p1-p2|^2 + (p1 x p2) * |p2-p3|^2 ) / (6 * Volume)
            # This is complex. 
            # Simpler: The angular radius r of the circumcircle of the spherical triangle.
            # The Voronoi vertex is at angular distance r from the sites.
            # We can find r by solving for the point on the normal n.
            # Let v = t * n.
            # |v - p1|^2 = |v|^2 - 2 t n.dot(p1) + 1
            # We want to minimize max_i |v - pi|^2? No, v is the vertex, it's fixed by the geometry.
            # The vertex is the unique point equidistant to p1, p2, p3.
            # t^2 - 2 t n.dot(p1) + 1 = t^2 - 2 t n.dot(p2) + 1
            # n.dot(p1) = n.dot(p2) = n.dot(p3) = h (height of triangle from origin)
            # So the equation is t^2 - 2 t h + 1 = 0.
            # t = h +/- sqrt(h^2 - 1).
            # For the Voronoi vertex to exist (real t), we need h^2 >= 1.
            # h = n.dot(p1). Since n is unit normal, h = cos(angle between n and p1).
            # Wait, if the triangle is acute (contains origin in its polar dual?), h > 0.
            # The Voronoi vertex is the one with smaller |t|? Or the one further out?
            # The Voronoi cell is bounded. The vertex is the one that is "outside" the triangle relative to the center?
            # Actually, for a covering, the "worst" point is the one farthest from the sites.
            # The Voronoi vertex is the center of the empty cap.
            # The distance from the site to the vertex is the covering radius for that cell.
            # Let's compute the angular distance.
            # cos(theta) = (1 - |v|^2) / (2 * |v| * 1) ? No.
            # Law of cosines on the spherical triangle?
            
            # Let's just compute the circumradius of the Euclidean triangle p1, p2, p3.
            # R_euclid = |p1-p2||p2-p3||p3-p1| / (4 * Area)
            # The angular radius theta satisfies sin(theta) = R_euclid / (2 * R_sphere) ? No.
            # In 3D, if the triangle is on the unit sphere, the circumcenter C is at distance d from origin.
            # R_euclid^2 = d^2 - 1 (if C is outside the sphere? No, C is on the normal).
            # |C - p1|^2 = d^2 - 2 d h + 1.
            # We know |C - p1| is the chord length.
            # Let's just find the max angular distance to the nearest point for a dense grid, but use the hull to guide optimization.
            pass
        
        # Fallback: Dense grid check for evaluation, but this is slow.
        # For speed, we'll just return a rough estimate or use the hull equations if we can interpret them.
        # Actually, let's just use the grid method for the final score, but use the hull for the gradient.
        
    except Exception:
        pass
    
    # For the purpose of this function in the optimization loop, we need a fast proxy.
    # The max circumradius of the hull facets is a good proxy.
    # Let's compute the max distance from origin to the facet planes?
    # No, the covering radius is the max distance from a point on the sphere to the nearest site.
    # This is equal to the max circumradius of the spherical Voronoi cells.
    # For a facet with normal n and distance d from origin (n.x = d), 
    # the circumcenter of the spherical triangle is along n.
    # The angular radius theta is given by cos(theta) = 1/sqrt(1 + (d^2 - 1)) ?
    
    # Let's just do a quick grid check here, it's N=130, grid 50x100=5000 points.
    # 5000 * 130 = 650k ops. Fast enough.
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    test_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Vectorized distance
    # points (N, 3), test_points (M, 3)
    # dists^2 = |p|^2 + |t|^2 - 2 p.t = 2 - 2 p.t
    dots = test_points @ points.T # (M, N)
    dists_sq = 2 - 2 * dots
    dists_sq = np.clip(dists_sq, 0, 4)
    min_dists_sq = np.min(dists_sq, axis=1)
    min_dists = np.sqrt(min_dists_sq)
    angular_dists = 2 * np.arcsin(min_dists / 2)
    return np.max(angular_dists) * 180 / np.pi

def _spectral_gradient(points):
    """
    Gradient of the potential energy U = sum_{i<j} 1/|pi-pj|^2
    This encourages uniform distribution.
    """
    N = len(points)
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-10: continue
            grad[i] += -2.0 * diff / (d2 * d2)
    return grad

def _voronoi_gradient(points, start_time, deadline):
    """
    Identify the worst Voronoi cell (largest circumradius) and push its vertices.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        vertices = hull.vertices
        
        # Find the facet with the largest circumradius
        # Approximate by finding the facet whose normal is farthest from the origin?
        # Or just compute the circumradius for each.
        
        max_rad = -1
        worst_simplex = None
        worst_center = None
        
        for simplex in simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Compute circumcenter
            # Using the formula for circumcenter of 3 points on unit sphere
            # C = (p1 + p2 + p3) / (1 + p1.p2 + p2.p3 + p3.p1 - p1.p2.p3) ? No.
            # Let's use the linear algebra solution.
            # C is equidistant to p1, p2, p3.
            # (C-p1).(C+p1) = 0 => C.C = C.p1
            # C.C = C.p2
            # C.C = C.p3
            # So C.p1 = C.p2 = C.p3.
            # C is in the span of p1, p2, p3.
            # C = a p1 + b p2 + c p3
            # This is a 3x3 system.
            
            A = np.array([
                [np.dot(p1, p1), np.dot(p1, p2), np.dot(p1, p3)],
                [np.dot(p2, p1), np.dot(p2, p2), np.dot(p2, p3)],
                [np.dot(p3, p1), np.dot(p3, p2), np.dot(p3, p3)]
            ])
            b = np.array([1, 1, 1]) # Since |pi|^2 = 1, C.pi = 1
            
            try:
                coeffs = np.linalg.solve(A, b)
                C = coeffs[0]*p1 + coeffs[1]*p2 + coeffs[2]*p3
                # The angular radius is the angle between C (normalized) and p1?
                # No, the vertex is C. The distance from C to p1 is the chord length.
                # Angular distance theta: cos(theta) = (C.p1) / (|C| * 1) = 1 / |C|
                # So theta = arccos(1 / |C|)
                C_norm = np.linalg.norm(C)
                if C_norm > 1e-10:
                    cos_theta = 1.0 / C_norm
                    if cos_theta > 1.0: cos_theta = 1.0
                    if cos_theta < -1.0: cos_theta = -1.0
                    theta = np.arccos(cos_theta)
                    if theta > max_rad:
                        max_rad = theta
                        worst_simplex = simplex
                        worst_center = C
            except np.linalg.LinAlgError:
                continue
        
        if worst_simplex is not None:
            # We want to shrink this cell.
            # Move the vertices of the worst simplex towards the center of the cell?
            # Or move them away from the "hole"?
            # The "hole" is at worst_center.
            # To shrink the cap, we should move the sites closer to the hole?
            # No, the sites are on the sphere. The hole is "outside" the triangle.
            # Moving sites towards the hole (along the great circle) will reduce the circumradius.
            
            grad = np.zeros_like(points)
            center_norm = np.linalg.norm(worst_center)
            if center_norm > 1e-10:
                dir_to_hole = worst_center / center_norm
                
                for idx in worst_simplex:
                    p = points[idx]
                    # Component of dir_to_hole tangent to the sphere at p
                    # Tangent vector = dir_to_hole - (dir_to_hole . p) p
                    proj = np.dot(dir_to_hole, p)
                    tangent = dir_to_hole - proj * p
                    # Move p in the direction of tangent
                    grad[idx] += tangent
            
            return grad
            
    except Exception:
        return np.zeros_like(points)
    
    return np.zeros_like(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = _compute_covering_radius(points)
    
    # Optimization loop
    lr_spectral = 0.005
    lr_voronoi = 0.02
    
    for it in range(5000):
        if not _time_ok(start_time, deadline):
            break
            
        # 1. Spectral Uniformity Step
        grad_spec = _spectral_gradient(points)
        points = points + lr_spectral * grad_spec
        points = _normalize(points)
        
        # 2. Voronoi Refinement Step
        if it % 5 == 0: # Every 5th iteration, do a Voronoi specific check
            grad_vor = _voronoi_gradient(points, start_time, deadline)
            if np.linalg.norm(grad_vor) > 1e-10:
                points = points + lr_voronoi * grad_vor
                points = _normalize(points)
                
                # Check if we improved
                current_score = _compute_covering_radius(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        # Periodic check
        if it % 100 == 0:
             current_score = _compute_covering_radius(points)
             if current_score < best_score:
                 best_score = current_score
                 best_points = points.copy()
    
    # Final check
    final_score = _compute_covering_radius(best_points)
    if final_score < best_score:
        best_points = points.copy()
        
    return best_points
