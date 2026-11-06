"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    """Normalize rows to unit length, handling zero vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum, over all points on the sphere,
    of the angular distance to the nearest point in pts.
    This equals the maximum circumradius of the facets of the convex hull
    of pts (when pts are on the unit sphere).
    """
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Check for coincident points
    # If any two points are too close, it's invalid per contract
    # But for scoring, we just compute the radius
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v
    # such that v . p_i >= 0 for all i.
    # This is equivalent to the origin not being in the interior of the convex hull.
    # If the origin is on the boundary or outside, the covering radius is >= 90 degrees.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # If hull computation fails, return a large value
        return 180.0
    
    # For each facet, compute the circumradius
    # The circumradius of a spherical triangle (or polygon) inscribed in the unit sphere
    # is the angular distance from the sphere center (origin) to the facet's circumcenter.
    # Actually, the covering radius is the maximum over all facets of the angular distance
    # from the origin to the facet plane, but only if the origin is on the "inside".
    # More precisely, for a facet with normal n (pointing outward from the convex hull),
    # the angular distance from the origin to the facet is arccos(n . c) where c is the
    # circumcenter of the facet on the sphere... 
    #
    # Actually, the correct formula: for a facet of the convex hull of points on the unit sphere,
    # the cap corresponding to that facet has its center at the normalized centroid of the facet
    # vertices (for a triangle, it's the circumcenter on the sphere). The angular radius of that
    # cap is the angle between the facet's circumcenter and any vertex.
    #
    # Let me use a more robust approach: for each facet, find the point on the sphere
    # farthest from the facet's vertices. This is the circumcenter of the spherical polygon.
    # For a spherical triangle, this is the circumcenter.
    #
    # A simpler approach: the covering radius is the maximum, over all facets, of
    # the angle between the origin and the facet, measured as the angular distance
    # from the origin to the closest point on the facet (when projected to the sphere).
    #
    # Actually, the standard result: the covering radius equals the maximum circumradius
    # of the facets of the convex hull. For a facet with vertices v_1, ..., v_k on the unit sphere,
    # the circumradius is the angular distance from the circumcenter (on the sphere) to any vertex.
    #
    # For a triangle on the sphere, the circumcenter is the normalized sum of the vertices
    # only if the triangle is acute. In general, we need to solve for it.
    #
    # Let me use a different approach: for each facet, the cap center is the point on the sphere
    # that maximizes the minimum angular distance to the facet's vertices. This is the spherical
    # circumcenter. For a triangle, it can be found by solving the linear system.
    #
    # Simpler: use the fact that for points on the unit sphere, the circumradius of a facet
    # is arccos(h) where h is the distance from the origin to the facet plane, IF the origin
    # is on the same side as the interior. But this gives the "inradius" of the cap, not the circumradius.
    #
    # Let me just compute it directly: for each facet, find the circumcenter by solving
    # |x - v_i|^2 = r^2 for all vertices, with |x| = 1.
    #
    # For a triangle with vertices a, b, c on the unit sphere:
    # The circumcenter x satisfies x.a = x.b = x.c = cos(r)
    # So x is in the direction of the solution to the linear system.
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        verts = pts[simplex]  # shape (k, 3) where k is typically 3
        
        # Find the circumcenter on the sphere
        # For a spherical simplex, the circumcenter is the normalized vector
        # that has equal dot product with all vertices.
        # Solve: x . v_i = c for all i, |x| = 1
        # This means x is in the span of the vertices, and x = A^T * w for some weights
        # such that A * x = c * 1
        
        # For a triangle (3 vertices), we can solve directly
        k = len(verts)
        if k == 3:
            a, b, c = verts[0], verts[1], verts[2]
            # The circumcenter is proportional to a cross b + b cross c + c cross a
            # Actually, let me solve the linear system properly.
            # x . a = x . b = x . c = cos(r)
            # Let x = alpha*a + beta*b + gamma*c
            # Then:
            # alpha*(a.a) + beta*(a.b) + gamma*(a.c) = cos(r)
            # alpha*(b.a) + beta*(b.b) + gamma*(b.c) = cos(r)
            # alpha*(c.a) + beta*(c.b) + gamma*(c.c) = cos(r)
            # And |x|^2 = 1
            #
            # Since a, b, c are unit vectors, a.a = b.b = c.c = 1
            # Let G be the Gram matrix: G[i,j] = v_i . v_j
            # Then G * w = cos(r) * 1, where w = [alpha, beta, gamma]
            # And |x|^2 = w^T G w = cos^2(r)
            # So cos(r) = sqrt(w^T G w)
            
            G = verts @ verts.T
            ones = np.ones(k)
            # Solve G * w = ones (this gives x . v_i = 1 for all i, which is impossible unless r=0)
            # Instead, we want to find the direction x such that x . v_i are all equal.
            # The vector x = sum(w_i * v_i) has x . v_j = sum_i w_i * G[i,j]
            # We want sum_i w_i * G[i,j] = c for all j, i.e., G^T w = c * 1
            # Since G is symmetric, G w = c * 1, so w = c * G^{-1} 1
            # The direction is G^{-1} 1, and c = 1 / sqrt((G^{-1}1)^T G (G^{-1}1)) = 1 / sqrt(1^T G^{-1} 1)
            
            try:
                inv_G = np.linalg.inv(G)
                w = inv_G @ ones
                denom = np.sqrt(ones @ inv_G @ ones)
                cos_r = 1.0 / denom
                if cos_r > 1.0:
                    cos_r = 1.0
                if cos_r < -1.0:
                    cos_r = -1.0
                r = np.arccos(cos_r)
            except np.linalg.LinAlgError:
                r = np.pi / 2  # fallback
        elif k == 4:
            # For a quadrilateral, the circumcenter might not exist on the sphere
            # in the same way. Use the triangle decomposition or just take the max
            # of the triangle circumradii.
            # For now, use the first three vertices as a triangle
            tri = verts[:3]
            G = tri @ tri.T
            ones3 = np.ones(3)
            try:
                inv_G = np.linalg.inv(G)
                w = inv_G @ ones3
                denom = np.sqrt(ones3 @ inv_G @ ones3)
                cos_r = 1.0 / denom
                if cos_r > 1.0:
                    cos_r = 1.0
                if cos_r < -1.0:
                    cos_r = -1.0
                r = np.arccos(cos_r)
            except np.linalg.LinAlgError:
                r = np.pi / 2
        else:
            r = np.pi / 2
        
        max_radius = max(max_radius, r)
    
    return np.degrees(max_radius)


def _golden_spiral(n, seed=0):
    """Generate points on the sphere using a Fibonacci spiral with optional perturbation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _perturb(pts, step_size, rng):
    """Add small random perturbations to points on the sphere."""
    # Generate random unit vectors for perturbation
    noise = rng.randn(*pts.shape)
    pts_new = pts + step_size * noise
    return _normalize(pts_new)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with golden spiral
    pts = _golden_spiral(N)
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(best_pts)
    
    # Riemannian gradient descent with focus on worst-case
    # The covering radius is non-smooth, so we use subgradient methods
    # Focus on the worst facet to drive improvement
    
    rng = np.random.RandomState(42)
    
    # Annealing schedule
    initial_step = 0.1
    final_step = 0.001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute covering radius and find worst point
        # The worst point is the circumcenter of the worst facet
        # For now, use a simple approach: find the point on the sphere
        # farthest from all configuration points
        
        # Find the worst point by sampling
        # The worst point is on the boundary of the Voronoi cell of the farthest point
        # Simplified: find the point that maximizes the minimum distance
        
        # Compute pairwise distances
        # For efficiency, use the fact that the worst point is near the circumcenter
        # of the worst facet
        
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_facet_radius = 0
        worst_facet_idx = 0
        for i, simplex in enumerate(hull.simplices):
            verts = pts[simplex]
            k = len(verts)
            if k >= 3:
                tri = verts[:3]
                G = tri @ tri.T
                ones3 = np.ones(3)
                try:
                    inv_G = np.linalg.inv(G)
                    w = inv_G @ ones3
                    denom = np.sqrt(ones3 @ inv_G @ ones3)
                    cos_r = 1.0 / denom
                    if cos_r > 1.0:
                        cos_r = 1.0
                    if cos_r < -1.0:
                        cos_r = -1.0
                    r = np.arccos(cos_r)
                except np.linalg.LinAlgError:
                    r = np.pi / 2
                if r > worst_facet_radius:
                    worst_facet_radius = r
                    worst_facet_idx = i
        
        # Get the worst facet vertices
        worst_simplex = hull.simplices[worst_facet_idx]
        worst_verts = pts[worst_simplex]
        
        # Find the circumcenter of the worst facet
        if len(worst_verts) >= 3:
            tri = worst_verts[:3]
            G = tri @ tri.T
            ones3 = np.ones(3)
            try:
                inv_G = np.linalg.inv(G)
                w = inv_G @ ones3
                circumcenter = w @ tri  # This is the circumcenter in R^3
                circumcenter = circumcenter / np.linalg.norm(circumcenter)
            except np.linalg.LinAlgError:
                circumcenter = worst_verts.mean(axis=0)
                circumcenter = circumcenter / np.linalg.norm(circumcenter)
        else:
            circumcenter = worst_verts.mean(axis=0)
            circumcenter = circumcenter / np.linalg.norm(circumcenter)
        
        # Move points that are "responsible" for the worst facet toward the circumcenter
        # The subgradient of the covering radius w.r.t. each point in the worst facet
        # points from the point toward the circumcenter
        
        # Annealing
        t = step / max(1, n_steps - 1)
        step_size = initial_step * (final_step / initial_step) ** t
        step_size = max(step_size, final_step)
        
        # Move the worst facet vertices toward the circumcenter
        for idx in worst_simplex:
            # Direction from point to circumcenter
            direction = circumcenter - pts[idx]
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            pts[idx] = pts[idx] + step_size * direction
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
        
        # Add small random perturbations to escape local minima
        if step % 10 == 0:
            noise = rng.randn(*pts.shape) * step_size * 0.1
            pts = _normalize(pts + noise)
        
        # Check improvement
        current_radius = _covering_radius_deg(pts)
        if current_radius < best_radius - 1e-9:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Final check
    if time.time() < deadline:
        final_radius = _covering_radius_deg(best_pts)
        if final_radius < best_radius:
            best_radius = final_radius
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
