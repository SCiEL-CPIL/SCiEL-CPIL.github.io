"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization: Lloyd's Algorithm on the Sphere
    # We iterate:
    # a) Compute Convex Hull of points
    # b) For each point, find its Voronoi region (dual to hull facets)
    #    Actually, Lloyd's on sphere: move point to the centroid of its spherical Voronoi cell.
    #    A simpler and robust approach for covering:
    #    Use a force-directed method where points repel each other.
    #    However, to minimize the *covering* radius (max distance to nearest point),
    #    we want points to spread out. The "worst" points are those far from the nearest neighbor.
    #    
    #    Let's use a standard Lloyd's iteration for spherical TSP/covering:
    #    1. Compute Delaunay triangulation (via ConvexHull in 3D).
    #    2. The centroid of the spherical Voronoi cell of point i is the normalized sum of its neighbors in the Delaunay graph?
    #    No, for covering, the "center" of the cell is the point that maximizes the distance to the cell boundary.
    #    This is the Chebyshev center of the Voronoi cell.
    #    
    #    A more efficient heuristic often used:
    #    Iterate:
    #    1. Compute distances between all pairs.
    #    2. Identify the "worst" covered area.
    #    
    #    Let's stick to a robust Lloyd variant:
    #    Move each point towards the centroid of its adjacent points in the Delaunay triangulation.
    #    This tends to equalize the area/distance distribution.
    
    def get_delaunay_neighbors(points):
        try:
            hull = ConvexHull(points)
            # hull.simplices contains triangles (i, j, k)
            # Build adjacency list
            adj = {i: set() for i in range(N)}
            for simplex in hull.simplices:
                i, j, k = simplex
                adj[i].add(j)
                adj[i].add(k)
                adj[j].add(i)
                adj[j].add(k)
                adj[k].add(i)
                adj[k].add(j)
            return adj
        except Exception:
            # If hull fails (e.g., points too close or degenerate), return empty
            return {i: set() for i in range(N)}

    def lloyd_step(points, alpha=0.5):
        adj = get_delaunay_neighbors(points)
        new_points = points.copy()
        
        for i in range(N):
            neighbors = list(adj[i])
            if not neighbors:
                continue
            
            # Centroid of neighbors on the sphere
            # Simple average and normalize
            centroid = np.sum(points[neighbors], axis=0)
            norm = np.linalg.norm(centroid)
            if norm < 1e-10:
                continue
            centroid = centroid / norm
            
            # Move towards centroid
            new_points[i] = (1 - alpha) * points[i] + alpha * centroid
            # Renormalize
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
            
        return new_points

    # Optimization loop
    best_points = points
    best_score = np.inf
    
    # We can also try a "Repulsion" method which is sometimes faster to converge for covering
    # But Lloyd is more stable. Let's run Lloyd for a few iterations.
    # Given 50 points, ConvexHull is fast.
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        points = lloyd_step(points, alpha=0.3)
        
        # Calculate current covering radius (approximate by checking max distance to nearest neighbor? No, that's packing.
        # Covering radius is max over all y on sphere of min dist to points.
        # This is equivalent to the max circumradius of the hull facets.
        # We can estimate this by checking the circumcenters of the hull facets.
        
        try:
            hull = ConvexHull(points)
            # For each facet (triangle), find the circumcenter.
            # The distance from the origin to the circumcenter is related to the covering radius?
            # Actually, the covering radius is the maximum distance from the origin to the circumcenter of any facet,
            # IF the circumcenter lies within the facet. If not, the max is on the edge/vertex.
            # For a "good" distribution, circumcenters should be inside.
            
            max_circumradius = 0.0
            # hull.equations gives plane equations a*x+b*y+c*z=d.
            # For unit sphere, the distance from origin to plane is |d|/sqrt(a^2+b^2+c^2).
            # Wait, the circumcenter of the triangle formed by points on the unit sphere
            # is NOT simply the normal to the plane.
            
            # Let's use a simpler proxy for score during optimization:
            # The "packing" radius (min distance between points) is correlated but not the same.
            # However, for optimization, we can just run Lloyd and check the final score.
            
            pass
        except Exception:
            pass
        
        # Check time and break if converged
        if it % 10 == 0:
             # Quick check: if points haven't moved much
             pass

    # Final refinement: Check if we can improve by "pushing" points away from each other
    # A simple gradient descent on the covering radius is hard.
    # Let's try a local search: for each point, try moving it slightly in random directions
    # to see if the max circumradius decreases.
    
    def calc_covering_radius(points):
        try:
            hull = ConvexHull(points)
            # The covering radius is the max over all facets of the distance from the origin
            # to the circumcenter of the facet, provided the circumcenter is inside the facet.
            # If the circumcenter is outside, the max distance in that facet is to a vertex or edge midpoint.
            # Actually, the function f(y) = min_i dist(y, p_i) is concave on each spherical triangle?
            # No.
            
            # A robust way: The covering radius is the smallest r such that spheres of radius r cover the sphere.
            # This is the max over y of min_i dist(y, p_i).
            # This maximum occurs at a circumcenter of a Delaunay triangle, or a midpoint of an edge, or a vertex.
            
            max_r = 0.0
            # Get all triangles from hull
            triangles = hull.simplices
            
            for t in triangles:
                p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]
                
                # Find circumcenter of triangle on sphere
                # Solve for x such that x.p1 = x.p2 = x.p3 = c, and |x|=1?
                # No, the circumcenter of the triangle on the sphere is the point on the sphere
                # equidistant to p1, p2, p3.
                # This is the intersection of the sphere and the plane defined by p1,p2,p3?
                # No, the plane p1,p2,p3 cuts the sphere. The center of the circle is the projection of the origin onto the plane.
                # The points on the sphere equidistant to p1,p2,p3 are the intersections of the sphere with the line
                # passing through the origin and the circumcenter of the planar triangle?
                
                # Let's compute the circumcenter of the planar triangle p1,p2,p3.
                # Let O be origin.
                # The circumcenter C_planar of the triangle in 3D space lies on the line passing through O and the normal to the plane?
                # No.
                
                # Standard method:
                # The circumcenter of the spherical triangle is the normalized vector sum of p1, p2, p3 weighted by cos of opposite sides?
                # Or simply: find the intersection of the sphere with the line defined by the cross products?
                
                # Let's use the property: The circumcenter C satisfies C.p1 = C.p2 = C.p3.
                # C is a linear combination of p1, p2, p3.
                # C = a*p1 + b*p2 + c*p3
                # C.p1 = C.p2 => a(1) + b(p2.p1) + c(p3.p1) = a(p1.p2) + b(1) + c(p3.p2)
                # This is a system of linear equations.
                
                # Simpler: Use the fact that the circumcenter is the normalized sum of the vertices
                # weighted by the sine of the opposite angles?
                
                # Let's just compute the circumcenter of the planar triangle and project it?
                # The circumcenter of the planar triangle is equidistant from p1,p2,p3 in Euclidean distance.
                # Let C_planar be that point.
                # The spherical circumcenter is C_sph = C_planar / |C_planar| ?
                # Let's verify.
                # If C_planar is equidistant from p1,p2,p3, then |C_planar - p1| = |C_planar - p2|.
                # |C_planar|^2 - 2 C_planar.p1 + |p1|^2 = |C_planar|^2 - 2 C_planar.p2 + |p2|^2
                # Since |p1|=|p2|=1, this implies C_planar.p1 = C_planar.p2.
                # So C_planar is equidistant from p1,p2,p3 in terms of dot product.
                # Therefore, C_sph = C_planar / |C_planar| is equidistant from p1,p2,p3 on the sphere.
                
                # Compute C_planar
                # Let u = p2 - p1, v = p3 - p1
                # C_planar = p1 + s*u + t*v
                # (C_planar - p1).u = |u|^2 / 2  => s|u|^2 + t(u.v) = |u|^2/2
                # (C_planar - p1).v = |v|^2 / 2  => s(u.v) + t|v|^2 = |v|^2/2
                
                u = p2 - p1
                v = p3 - p1
                uu = np.dot(u, u)
                vv = np.dot(v, v)
                uv = np.dot(u, v)
                
                det = uu * vv - uv * uv
                if abs(det) < 1e-10:
                    continue # Degenerate
                
                s = (0.5 * uu * vv - 0.5 * uv * vv) / det
                t = (0.5 * vv * uu - 0.5 * uu * uv) / det
                
                c_planar = p1 + s * u + t * v
                c_norm = np.linalg.norm(c_planar)
                if c_norm < 1e-10:
                    continue
                c_sph = c_planar / c_norm
                
                # Check if c_sph is inside the spherical triangle
                # A point is inside if it can be expressed as a positive combination of the vertices?
                # Or check angles.
                # Simpler: The circumcenter is inside if the triangle is acute.
                # We can check if the distance from c_sph to any vertex is less than 90 degrees?
                # No, that's for packing.
                
                # For covering radius, the candidate points are the circumcenters.
                # The covering radius is the max over all y of min dist(y, p_i).
                # The maximum of this function occurs at a circumcenter of a Delaunay triangle.
                
                dist_to_p1 = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                # The distance from c_sph to the nearest point in the triangle's Voronoi cell boundary?
                # No, c_sph is the center. The distance to the nearest site is the circumradius.
                
                # Is c_sph the actual maximum?
                # Only if c_sph is inside the triangle.
                # How to check if c_sph is inside the spherical triangle?
                # Project c_sph onto the plane of the triangle?
                
                # Let's just compute the distance to the nearest point for this c_sph.
                d1 = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_sph, p2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_sph, p3), -1.0, 1.0))
                min_d = min(d1, d2, d3)
                
                # If c_sph is outside, the max might be on the edge.
                # But for a "good" configuration, circumcenters are inside.
                # We can check if the angle between c_sph and the normal to the edge is correct.
                
                # To be safe, we can also check edge midpoints.
                # But let's assume circumcenters are the critical points for now.
                
                if min_d > max_r:
                    max_r = min_d
                    
            # Also check edges?
            # For N=50, checking all edges is O(N^2) which is fine.
            # But hull.simplices gives edges implicitly.
            
            return np.degrees(max_r)
            
        except Exception:
            return 180.0 # Invalid

    # Run a few iterations of Lloyd and check score
    # We will run Lloyd, then do a local search to fine-tune.
    
    current_points = points
    best_score = calc_covering_radius(current_points)
    
    for it in range(100):
        if time.time() > deadline:
            break
            
        current_points = lloyd_step(current_points, alpha=0.2)
        score = calc_covering_radius(current_points)
        
        if score < best_score:
            best_score = score
            best_points = current_points.copy()
            
        # If we've converged, break
        # Check movement
        if it > 0:
             pass # We don't have prev points stored easily in this loop structure without extra var

    # Local Search: Perturb points to improve
    # Try moving each point in the direction that reduces the max circumradius.
    # This is complex. Instead, try random perturbations and keep if better.
    
    for it in range(50):
        if time.time() > deadline:
            break
             
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try 20 random directions
            for _ in range(20):
                # Random unit vector
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                
                # Move point i slightly
                eps = 0.05
                new_p = best_points[i] + eps * rand_vec
                new_p /= np.linalg.norm(new_p)
                
                old_p = best_points[i]
                best_points[i] = new_p
                
                new_score = calc_covering_radius(best_points)
                
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_points[i] = old_p
                    
        if not improved:
            break

    # Final check and return
    # Ensure points are normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
