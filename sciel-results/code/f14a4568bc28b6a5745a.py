"""Repulsive force-based optimization for 50-point sphere covering."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 50

# Known best covering radius (Hardin, Sloane, Smith 1994) in radians
_KNOWN_BEST_RAD = np.radians(18.3000226)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n < 1e-15] = 1.0
    return pts / n


def _covering_radius_rad(pts):
    """Compute covering radius in radians. Returns inf if invalid (hemisphere, etc.)."""
    pts = _normalize(pts)
    # Check not in a closed hemisphere: origin must be in interior of convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    # Check if origin is inside the convex hull (not in a closed hemisphere)
    # Use the convex hull: if all facets are on one side, origin is outside
    # A point is in a closed hemisphere iff there exists a direction u such that u·p >= 0 for all p
    # Equivalent: origin is in the convex hull of pts
    # Check via: for each facet, the origin should be on the "inside" (negative side of outward normal)
    # Actually simpler: check if 0 is in the convex hull using scipy
    from scipy.spatial import ConvexHull as _CH
    # Use linear programming or check via the hull
    # The origin is in the convex hull iff for every facet, origin is on the interior side
    # ConvexHull gives equations: normal·x + offset <= 0 for interior
    # So origin is inside iff offset >= 0 for all facets (since normal·0 + offset = offset)
    # Wait, scipy convention: equations[i] = [n, c] means n·x + c <= 0
    # Origin inside iff c >= 0 for all i? No: n·0 + c = c <= 0 means origin satisfies the constraint.
    # But we need origin strictly inside (not on boundary) to avoid closed hemisphere issue.
    # Actually the task says "configurations inside a closed hemisphere" are invalid.
    # So we need origin in the INTERIOR of the convex hull.
    # If any facet has c == 0 (within tolerance), origin is on the boundary -> closed hemisphere.
    min_offset = np.min(hull.equations[:, -1])
    if min_offset <= 1e-9:
        return np.inf
    # Now compute covering radius as max circumradius of facets
    # For each facet (triangle), compute the circumradius and check the cap is empty
    # The covering radius is the max over all points on the sphere of the distance to nearest point.
    # This equals the max circumradius of the spherical Voronoi cells, which corresponds to
    # the circumcircles of the Delaunay triangles on the sphere.
    # For a triangle on the unit sphere with vertices a, b, c, the circumradius (angular)
    # is the angular distance from the circumcenter to each vertex.
    # The circumcenter of a spherical triangle is the normalized cross product approach,
    # or we can compute it as the point equidistant from all three vertices.
    # For three points on the unit sphere, the circumcenter is along the direction:
    #   (b - a) x (c - a)  ... no, that's the normal.
    # Actually the circumcenter of three points on a sphere is found by:
    #   Find the plane through the three points, intersect with the sphere.
    #   The circumcenter is the center of that circle projected to the sphere? No.
    #   The circumcenter on the sphere is the point equidistant (angularly) from all three.
    #   This is the normalized version of: a + b + c adjusted... 
    #   Actually for three points on the unit sphere, the circumcenter is:
    #   normalize( (|b|^2(c-a) + |c|^2(a-b) + |a|^2(b-c) ) / something )
    #   Simpler: the circumcenter is the intersection of the sphere with the line
    #   through the origin in the direction of the normal to the plane? No.
    #   Let me use: the circumcenter of triangle abc on unit sphere satisfies
    #   |circ - a| = |circ - b| = |circ - c| and |circ| = 1.
    #   This gives: circ·a = circ·b = circ·c = cos(theta) for some theta.
    #   So circ is in the span of a,b,c and satisfies the equalities.
    #   circ = alpha*a + beta*b + gamma*c, with circ·a = circ·b = circ·c.
    #   This is a linear system. Let's solve it.
    
    covering_rad = 0.0
    n_pts = len(pts)
    # For each Delaunay triangle (from the convex hull facets), compute circumradius
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Solve for circumcenter: find x = alpha*a + beta*b + gamma*c such that
        # x·a = x·b = x·c and |x| = 1
        # x·a = x·b => alpha(a·a) + beta(b·a) + gamma(c·a) = alpha(a·b) + beta(b·b) + gamma(c·b)
        # => alpha(1 - a·b) + beta(a·b - 1) + gamma(a·c - b·c) = 0
        # x·a = x·c => alpha(1 - a·c) + beta(b·a - c·a) + gamma(a·c - 1) = 0
        # Let's set up the system properly.
        # We want x such that x·a = x·b = x·c. Let this common value be t.
        # x = alpha*a + beta*b + gamma*c
        # x·a = alpha + beta*(b·a) + gamma*(c·a) = t
        # x·b = alpha*(a·b) + beta + gamma*(c·b) = t
        # x·c = alpha*(a·c) + beta*(b·c) + gamma = t
        # Subtracting: x·a - x·b = 0 and x·a - x·c = 0
        # (alpha + beta*ba + gamma*ca) - (alpha*ab + beta + gamma*cb) = 0
        # alpha(1-ab) + beta(ba-1) + gamma(ca-cb) = 0
        # Note ba = ab, so: alpha(1-ab) - beta(1-ab) + gamma(ca-cb) = 0
        # (alpha - beta)(1-ab) + gamma(ca-cb) = 0
        # Similarly: (alpha - gamma)(1-ac) + beta(ba-ca) = 0
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        # System:
        # (1-ab)(alpha - beta) + (ac-bc)*gamma = 0
        # (1-ac)(alpha - gamma) + (ab-ac)*beta = 0
        # Let u = alpha - beta, v = alpha - gamma
        # u(1-ab) + (ac-bc)*gamma = 0  ... but gamma = alpha - v
        # This is getting complex. Let me just use a direct approach.
        
        # Alternative: the circumcenter is the normalized vector:
        # x = (|b|^2 * (c - a) + |c|^2 * (a - b) + |a|^2 * (b - c)) ... no
        # For points on unit sphere, a known formula: the circumcenter is
        # normalize of the solution to the linear system.
        # Let me just solve the 3x3 system directly.
        # x = alpha*a + beta*b + gamma*c
        # Constraints: x·a = x·b, x·a = x·c
        # Matrix form:
        # [1, ab, ac] [alpha, beta, gamma]^T = t  (x·a = t)
        # [ab, 1, bc] [alpha, beta, gamma]^T = t  (x·b = t)
        # [ac, bc, 1] [alpha, beta, gamma]^T = t  (x·c = t)
        # Subtract row1 - row2: [1-ab, ab-1, ac-bc] · [alpha,beta,gamma] = 0
        # Subtract row1 - row3: [1-ac, ab-ac, ac-1] · [alpha,beta,gamma] = 0
        # Plus normalization: |x|^2 = 1, but let's first find the direction.
        # Two equations, three unknowns. We need one more condition.
        # The circumcenter should be "between" the points, so let's use the condition
        # that it's the minimum norm solution or use the fact that for a triangle
        # on a sphere, the circumcenter lies in the plane of the triangle... 
        # Actually, let me just use the fact that the circumcenter of three points
        # on the unit sphere can be found as:
        #   n = (b - a) x (c - a)  (normal to plane)
        #   The circumcenter is along the direction from the origin toward the
        #   circumcenter of the planar triangle, projected.
        #   For three points on a sphere, the circumcenter (on the sphere) is:
        #   normalize( a + b + c ) only for equilateral... no.
        
        # Let me use the direct linear algebra approach with 3 equations:
        # x·a = x·b, x·a = x·c, and x is in the plane of a,b,c (which is automatic
        # since x = alpha*a + beta*b + gamma*c).
        # Actually I'll just use the formula: for three unit vectors, the circumcenter
        # direction is proportional to:
        #   (b - a) x (c - a)  is the normal. The circumcenter is NOT along the normal.
        #   
        # Let me just solve it numerically with a small system.
        # We have: M @ [alpha, beta, gamma]^T = 0 where M is 2x3, plus we want
        # the solution that minimizes |x|^2 subject to the constraints? No.
        # 
        # Actually, the simplest: the circumcenter of a spherical triangle is the
        # pole of the plane containing the three vertices? No, that's the incenter
        # of the antipodal triangle.
        #
        # Let me just use: for three points on the unit sphere, the circumcenter
        # is the normalized vector:
        #   v = (1/(1-(ab)^2)) * [ (1-ac)*a + (ab - ac)*b + ... ]
        # This is getting too complicated. Let me use a different approach.
        
        # The circumradius of a spherical triangle with side lengths a,b,c (angular)
        # can be computed via the spherical law of cosines and circumradius formula.
        # But an easier approach: the covering radius is the max over all points on
        # the sphere. The worst point is at the circumcenter of some Delaunay triangle
        # (or at infinity if the triangle is obtuse, but on a sphere that doesn't happen
        # the same way). 
        #
        # For three points on the unit sphere, the circumcenter (the point on the sphere
        # equidistant from all three) can be computed as:
        #   Find the plane through a, b, c: n = (b-a) x (c-a), d = n·a
        #   The circumcenter of the planar triangle projected to the sphere...
        #   Actually, the set of points on the sphere equidistant from a and b is the
        #   great circle through the midpoint direction. The circumcenter is where three
        #   such great circles meet.
        #
        # Let me just compute it as: the circumcenter direction is
        #   normalize( (b x c) * |a|^2 + (c x a) * |b|^2 + (a x b) * |c|^2 )
        # Wait, that's the formula for the circumcenter of a triangle in 3D:
        #   For a triangle with vertices A, B, C, the circumcenter is:
        #   O = (|A|^2 (B x C) + |B|^2 (C x A) + |C|^2 (A x B)) / (2 * A · (B x C))
        #   But this is for the circumcenter in 3D space, not on the sphere.
        #   For points on the unit sphere, the circumcenter ON the sphere is different.
        
        # OK let me just use the fact that the circumcenter on the sphere of triangle abc
        # is the normalized version of the solution to:
        #   x · (a - b) = 0  (equidistant from a and b)
        #   x · (a - c) = 0  (equidistant from a and c)
        #   |x| = 1
        # So x is perpendicular to (a-b) and (a-c), meaning x is parallel to (a-b) x (a-c).
        # But that's the normal to the plane! That can't be right for the circumcenter.
        # Wait: x·a = x·b means x·(a-b) = 0. Yes! So x is perpendicular to (a-b).
        # And x·(a-c) = 0. So x is perpendicular to both (a-b) and (a-c), hence
        # x is parallel to (a-b) x (a-c). But (a-b) x (a-c) is the normal to the plane
        # of the triangle. So the circumcenter on the sphere is along the normal to the
        # plane? That doesn't seem right for a general triangle.
        #
        # Wait, let me reconsider. x·a = x·b => x·(a-b) = 0. Yes.
        # x·a = x·c => x·(a-c) = 0. Yes.
        # So x is perpendicular to both (a-b) and (a-c), so x ∝ (a-b) × (a-c).
        # But (a-b) × (a-c) is the normal to the plane of the triangle.
        # So the circumcenter on the sphere IS along the normal to the plane of the triangle!
        # Let me verify: for an equilateral triangle centered on the z-axis, the normal
        # is along z, and the circumcenter should be at the "top" which is along z. Yes!
        # For a right triangle on the sphere... let's think. If a, b, c form a right
        # triangle on the sphere, the circumcenter should be at the midpoint of the
        # hypotenuse (for planar right triangles). On the sphere, the "midpoint" of the
        # hypotenuse arc is not along the normal. Hmm.
        #
        # Actually wait. I think I made an error. x·a = x·b does NOT mean x is perpendicular
        # to (a-b) in the sense I stated. x·a = x·b => x·a - x·b = 0 => x·(a-b) = 0. Yes,
        # that's correct. So x IS perpendicular to (a-b). And similarly for (a-c).
        # So x is along (a-b) × (a-c), the normal to the plane.
        #
        # Let me verify with a simple case. Take a = (1,0,0), b = (0,1,0), c = (0,0,1).
        # (a-b) = (1,-1,0), (a-c) = (1,0,-1). (a-b)×(a-c) = ((-1)(-1)-0*0, 0*1-1*(-1), 1*0-(-1)*1)
        # = (1, 1, 1). Normalized: (1,1,1)/sqrt(3).
        # Distance from (1,1,1)/sqrt(3) to a=(1,0,0): cos(theta) = 1/sqrt(3), theta = arccos(1/sqrt(3)) ≈ 54.7°.
        # Distance to b: same. Distance to c: same. So yes, it's equidistant! 
        # And the angular distance is arccos(1/sqrt(3)) ≈ 54.74°.
        # Is this the circumradius? The covering radius would be this if this is the worst triangle.
        # Seems reasonable.
        
        # So the circumcenter is along the normal to the plane of the triangle.
        # But we need to check that the cap is empty (no other point is closer to the
        # circumcenter than the triangle's vertices). For the covering radius, we only
        # need to consider the Delaunay triangles (from the convex hull), and for each,
        # the circumcenter should be checked to ensure no other point is inside the cap.
        # However, for the convex hull facets, the Delaunay condition on the sphere
        # ensures that the circumcenter's cap contains no other points (for the "empty
        # circumcircle" property). So we can just compute the circumradius for each facet.
        
        # Compute the normal to the plane of the triangle
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        # The circumcenter is either +n or -n (normalized). We want the one that is
        # "outside" the triangle, i.e., on the same side as the origin looking outward.
        # Actually, the circumcenter on the sphere should be the one that is farther
        # from the triangle's interior. Since the triangle is on the convex hull,
        # the outward normal points away from the origin. The circumcenter should be
        # on the "far side" of the triangle from the origin, i.e., in the direction
        # of the outward normal.
        # The outward normal of the convex hull facet points away from the interior
        # (away from the origin). So the circumcenter is along the outward normal.
        # But wait: the circumcenter is equidistant from a, b, c. Both +n and -n are
        # equidistant. We want the one that gives the LARGER distance (the one on the
        # opposite side of the triangle from the origin), because that's where the
        # "empty cap" is.
        # Actually, for the covering radius, we want the point on the sphere that is
        # farthest from all three vertices. That would be the antipode of the circumcenter
        # if the circumcenter is on the "near" side. Let me think again.
        #
        # The circumcenter is the point equidistant from a, b, c. There are two such points
        # on the sphere: +n_normalized and -n_normalized. One is closer to the triangle
        # (smaller angular distance) and one is farther. The covering radius for this
        # triangle is the angular distance to the FARTHER circumcenter, because that's
        # where a point could be that is far from all three vertices.
        # But wait, we also need the cap to be empty. If the farther circumcenter has
        # other points in its cap, then it's not a valid "worst point" for the covering.
        #
        # For the convex hull, the Delaunay triangulation on the sphere ensures that
        # the circumcenter of each facet has an empty cap (no other points in the cap).
        # So we just need to compute the angular distance from the circumcenter to the
        # vertices, and take the max over all facets.
        #
        # The angular distance from the circumcenter to a vertex: if the circumcenter is
        # at direction n (unit), and a vertex is at a (unit), the angular distance is
        # arccos(n·a).
        #
        # For the outward normal n: n·a should be positive (since n points outward and
        # a is on the hull). The angular distance is arccos(n·a).
        # For the inward normal -n: (-n)·a = -n·a, which is negative, so the angular
        # distance is arccos(-n·a) = pi - arccos(n·a), which is larger.
        #
        # So the farther circumcenter is -n (the inward normal), and the angular distance
        # is arccos(-n·a) = pi - arccos(n·a).
        # But wait, that would always be > 90° for hull facets, which seems too large.
        # Let me reconsider.
        #
        # Actually, I think the correct circumcenter for the covering radius is the one
        # that is on the "outside" of the triangle (away from the origin), which is +n
        # (the outward normal). The angular distance from +n to a is arccos(n·a).
        # For a small triangle near the "top" of the sphere, n ≈ (0,0,1) and a ≈ (0,0,1),
        # so n·a ≈ 1 and the angular distance is small. That makes sense.
        #
        # But is the cap empty? The cap centered at +n with angular radius arccos(n·a)
        # contains a, b, c on its boundary. For it to be empty (no other points inside),
        # we need the Delaunay condition. For convex hull facets, this should hold.
        #
        # So the covering radius is the max over all facets of arccos(n·a) where n is
        # the outward normal of the facet.
        
        # Check that n points outward (away from origin). For a convex hull, the
        # outward normal should satisfy n·a > 0 (since a is on the hull and n points out).
        if np.dot(n, a) <= 0:
            n = -n
        
        # Angular distance from circumcenter (n) to vertex a
        cos_theta = np.clip(np.dot(n, a), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta > covering_rad:
            covering_rad = theta
    
    return covering_rad


def _repulsive_forces(pts, strength=1.0, power=2.0):
    """Compute repulsive forces on each point based on nearest neighbors."""
    n = len(pts)
    forces = np.zeros_like(pts)
    # Compute pairwise distances (angular)
    # For efficiency, use dot products
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    # Angular distance
    ang_dist = np.arccos(dots)
    # Set diagonal to 0
    np.fill_diagonal(ang_dist, 0.0)
    
    # For each point, compute force from others
    # Force is repulsive, stronger for closer points
    # F_ij = strength * (1/d_ij)^power * (p_i - p_j)/|p_i - p_j| for d_ij < some cutoff
    # But we want points to move toward empty regions, so the force should push
    # points away from their neighbors.
    
    for i in range(n):
        # Find nearest neighbors
        dists = ang_dist[i]
        # Sort by distance
        order = np.argsort(dists)
        # Use all points (or top k) for the force
        for j in order:
            if i == j:
                continue
            d = dists[j]
            if d < 1e-10:
                continue
            # Repulsive force: push i away from j
            # Scale by 1/d^power to make close points repel more strongly
            # Also scale by how "dense" the local region is
            w = strength / (d ** power)
            # Direction: from j to i (pushing i away from j)
            direction = pts[i] - pts[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction = direction / norm_dir
            # Project force onto the tangent plane at pts[i]
            # The tangent plane is perpendicular to pts[i]
            tangent = direction - np.dot(direction, pts[i]) * pts[i]
            forces[i] += w * tangent
    return forces


def _optimize_repulsive(start_pts, max_time=100.0, initial_step=0.01):
    """Optimize using repulsive force dynamics."""
    t0 = time.time()
    pts = start_pts.copy()
    best_pts = pts.copy()
    best_rad = _covering_radius_rad(pts)
    step = initial_step
    n = len(pts)
    
    while time.time() - t0 < max_time:
        # Compute forces
        forces = _repulsive_forces(pts, strength=1.0, power=2.0)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms < 1e-15] = 1.0
        forces = forces / force_norms
        
        # Move points
        new_pts = pts + step * forces
        new_pts = _normalize(new_pts)
        
        # Evaluate
        new_rad = _covering_radius_rad(new_pts)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = new_pts.copy()
            step = min(step * 1.05, 0.05)  # Increase step if improving
        else:
            step *= 0.9  # Decrease step if not improving
        
        pts = new_pts
        
        # Early stop if we've reached the known best
        if best_rad <= _KNOWN_BEST_RAD * 1.001:
            break
    
    return best_pts, best_rad


def _random_start(seed):
    """Generate a random starting configuration."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _normalize(pts)


def _fibonacci_start():
    """Fibonacci spiral starting configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit
    
    # Try multiple random starts with repulsive optimization
    best_pts = _fibonacci_start()
    best_rad = _covering_radius_rad(best_pts)
    
    # First, optimize from the Fibonacci start
    pts, rad = _optimize_repulsive(best_pts, max_time=30.0)
    if rad < best_rad:
        best_rad = rad
        best_pts = pts.copy()
    
    # Try random starts
    seed = 42
    while time.time() < deadline:
        start = _random_start(seed)
        seed += 1
        pts, rad = _optimize_repulsive(start, max_time=20.0)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
            if best_rad <= _KNOWN_BEST_RAD * 1.001:
                break
    
    return _normalize(best_pts)
