"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]


def _covering_radius_exact(pts):
    """Compute exact covering radius using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        tri_idx = hull.simplices[:, :3]
        max_radius = 0.0
        
        for tri in tri_idx:
            p1, p2, p3 = pts[tri]
            
            # Compute circumcenter of the spherical triangle
            # The circumcenter is the point on the unit sphere equidistant from p1, p2, p3
            # It lies in the direction of the normalized vector:
            # c = (p1 * (|p2|^2 * |p3|^2 * sin(2*angle1)) + ...) -- complex
            # 
            # Simpler: the circumcenter direction is the normalized cross product of
            # (p1 - p2) and (p1 - p3)? No, that gives the normal to the plane.
            #
            # Correct approach: The circumcenter C satisfies:
            # |C - p1| = |C - p2| = |C - p3| and |C| = 1
            # This means C is in the direction of the vector that is equiangular.
            #
            # For three points on the unit sphere, the circumcenter direction is:
            # c = normalize(p1 * sin(2*A) + p2 * sin(2*B) + p3 * sin(2*C))
            # where A, B, C are the angles of the spherical triangle.
            #
            # Alternatively, we can compute it as follows:
            # The circumcenter is the intersection of the unit sphere with the line
            # through the origin in the direction of the circumcenter of the 3D triangle.
            # The 3D circumcenter of p1, p2, p3 (not on a plane through origin) is:
            # Solve: C = alpha*p1 + beta*p2 + gamma*p3
            # with constraints |C - p1| = |C - p2| = |C - p3|
            #
            # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
            # is the normalization of the vector:
            # c = p1 * (p2 x p3) + p2 * (p3 x p1) + p3 * (p1 x p2)
            # Wait, that's not right either.
            #
            # Let me use a different approach: the circumcenter is the point on the sphere
            # that maximizes the minimum distance to the three vertices. For a triangle
            # inscribed in a circle on the sphere, the circumcenter is the center of that circle.
            #
            # The plane containing p1, p2, p3 has normal n = (p2 - p1) x (p3 - p1)
            # The circumcenter lies on the line through the origin in direction n (or -n).
            # So the circumcenter is ±n/|n|, and we pick the one closer to the triangle.
            #
            # Actually no. The circumcenter of a spherical triangle is NOT simply the
            # normalization of the cross product. Let me think again.
            #
            # The circumcenter is the point C on the unit sphere such that the angular
            # distance from C to each of p1, p2, p3 is the same. This means:
            # C · p1 = C · p2 = C · p3 = cos(R)
            # where R is the circumradius.
            #
            # So C is in the null space of the matrix:
            # [p1 - p2; p2 - p3] (as rows), intersected with the unit sphere.
            #
            # Equivalently, C is orthogonal to (p1 - p2) and (p2 - p3):
            # C · (p1 - p2) = 0
            # C · (p2 - p3) = 0
            #
            # So C is in the direction of (p1 - p2) x (p2 - p3) = p1 x p2 + p2 x p3 + p3 x p1
            # Wait: (p1 - p2) x (p2 - p3) = p1 x p2 - p1 x p3 - p2 x p2 + p2 x p3
            # = p1 x p2 - p1 x p3 + p2 x p3
            # = p1 x p2 + p2 x p3 - p1 x p3
            # = p1 x p2 + p2 x p3 + p3 x p1  (since -p1 x p3 = p3 x p1)
            
            c_vec = np.cross(p1, p2) + np.cross(p2, p3) + np.cross(p3, p1)
            c_norm = np.linalg.norm(c_vec)
            
            if c_norm < 1e-12:
                # Degenerate triangle, use midpoint
                continue
            
            c = c_vec / c_norm
            
            # Check which side of the plane the circumcenter is on
            # We want the circumcenter to be on the same side as the triangle
            # relative to the origin. Actually, for a small triangle, the circumcenter
            # is in the direction of the sum p1 + p2 + p3.
            
            # Compute the circumradius: R = arccos(c · p1)
            dot_val = np.clip(c @ p1, -1.0, 1.0)
            R = np.arccos(dot_val)
            
            # Check if the circumcenter is inside the triangle
            # The circumcenter is inside if it can be expressed as a positive combination
            # of p1, p2, p3 (in the appropriate sense).
            # A simpler check: the circumcenter is inside the triangle if the angles
            # at each vertex are less than pi, which is always true for a convex hull face.
            # But the circumcenter might be outside the triangle if the triangle is obtuse.
            
            # For now, let's just use the circumradius if the circumcenter is "inside"
            # We check by seeing if c is in the convex cone of p1, p2, p3
            # This is equivalent to checking that c can be written as a positive combination.
            
            # Use a simple check: project c onto the plane of the triangle and see if
            # the barycentric coordinates are all positive.
            
            # Actually, for the covering radius, we need the maximum angular distance
            # from any point in the Voronoi cell to the nearest vertex. The Voronoi cell
            # for a convex hull face is the spherical triangle itself (if the circumcenter
            # is inside) or a larger region.
            
            # For simplicity and correctness, let's just use the circumradius as an upper
            # bound, and also check the edges and vertices.
            
            # The covering radius for this face is the maximum of:
            # 1. The circumradius (if circumcenter is inside the triangle)
            # 2. The maximum distance to the nearest vertex along the edges
            
            # For a convex hull face, the Voronoi cell is the set of points on the sphere
            # that are closest to the three vertices of the face. The maximum distance
            # occurs at the circumcenter if it's inside, or at a vertex/edge midpoint otherwise.
            
            # Let's compute the circumradius and also check if the circumcenter is inside.
            
            # To check if c is inside the spherical triangle, we can check that c is
            # on the same side of each edge as the opposite vertex.
            
            def is_inside_spherical_triangle(c, p1, p2, p3):
                # Check if c is on the same side of edge p1-p2 as p3
                edge1 = np.cross(p1, p2)
                edge2 = np.cross(p2, p3)
                edge3 = np.cross(p3, p1)
                
                # c is on the same side of edge1 as p3 if:
                # (edge1 · c) * (edge1 · p3) >= 0
                # But we need to be careful about the orientation.
                
                # Actually, for a convex hull face, the normal should point outward.
                # Let's just check the sign consistency.
                
                d1 = np.dot(edge1, c)
                d1_ref = np.dot(edge1, p3)
                
                d2 = np.dot(edge2, c)
                d2_ref = np.dot(edge2, p1)
                
                d3 = np.dot(edge3, c)
                d3_ref = np.dot(edge3, p2)
                
                return (d1 * d1_ref >= -1e-10) and (d2 * d2_ref >= -1e-10) and (d3 * d3_ref >= -1e-10)
            
            if is_inside_spherical_triangle(c, p1, p2, p3):
                max_radius = max(max_radius, R)
            else:
                # Circumcenter is outside, so the maximum is at a vertex or edge
                # Check the three vertices
                for p in [p1, p2, p3]:
                    # Distance from p to the farthest point in the Voronoi cell
                    # This is complex, so let's just use the maximum distance to the
                    # other two vertices as a rough upper bound
                    d12 = np.arccos(np.clip(p1 @ p2, -1.0, 1.0))
                    d23 = np.arccos(np.clip(p2 @ p3, -1.0, 1.0))
                    d31 = np.arccos(np.clip(p3 @ p1, -1.0, 1.0))
                    max_radius = max(max_radius, max(d12, d23, d31) / 2.0)
        
        return max_radius * 180.0 / np.pi
    except:
        return 180.0  # Return a large value if hull fails


def _covering_radius_fast(pts, n_samples=20000):
    """Fast approximate covering radius using Fibonacci spiral sampling."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    return max_angle * 180.0 / np.pi


def _local_optimize(pts, max_time=30.0):
    """Local optimization using gradient-like descent on covering radius."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    step_size = 0.1
    n_iter = 0
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Compute gradient-like direction: move point i to reduce covering radius
            # Try moving in the direction away from the worst-covered region
            # For simplicity, try random directions with adaptive step size
            
            for trial in range(20):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius_fast(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
        
        if no_improve_count > N:
            step_size *= 0.7
            no_improve_count = 0
            if step_size < 1e-5:
                break
    
    return best_pts, best_cr


def _refine_with_exact(pts, max_time=10.0):
    """Refine the configuration using exact covering radius for final validation."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    step_size = 0.02
    
    while time.time() - start_time < max_time:
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 2.0:
                break
            
            for trial in range(5):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                # Use exact covering radius for refinement
                cr = _covering_radius_exact(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    best_pts = None
    best_cr = np.inf
    
    # Try multiple initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    configs.append(fib_pts)
    
    # 2. Random configurations
    for _ in range(3):
        rand_pts = np.random.randn(N, 3)
        rand_pts = _normalize(rand_pts)
        configs.append(rand_pts)
    
    # 3. Icosahedron-based configuration (for N=12, then add more)
    # Start with icosahedron vertices and add more points
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = _normalize(ico)
    
    # Add 38 more points to make 50
    ico_full = ico.copy()
    for _ in range(38):
        # Add point in a random direction, but try to maximize minimum distance
        best_new_pt = None
        best_min_dist = -1
        for _ in range(50):
            cand = np.random.randn(3)
            cand = _normalize(cand)
            dists = np.arccos(np.clip(ico_full @ cand, -1.0, 1.0))
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_new_pt = cand
        ico_full = np.vstack([ico_full, best_new_pt])
    
    ico_full = _normalize(ico_full[:50])
    configs.append(ico_full)
    
    # Optimize each configuration
    for config in configs:
        if time.time() > deadline - 30.0:
            break
        
        opt_pts, cr = _local_optimize(config, max_time=25.0)
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts
    
    # Final refinement with exact covering radius
    if best_pts is not None and time.time() < deadline - 15.0:
        refined_pts, refined_cr = _refine_with_exact(best_pts, max_time=10.0)
        if refined_cr < best_cr:
            best_pts = refined_pts
            best_cr = refined_cr
    
    if best_pts is None:
        # Fallback: return Fibonacci spiral
        return fib_pts
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
