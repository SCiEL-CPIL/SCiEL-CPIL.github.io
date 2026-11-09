"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools

def _cover_radius(pts):
    """Compute the covering radius in radians.
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of that facet (for points on the unit sphere).
    """
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Check for coincident points
    if np.any(np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2) < 1e-12):
        # Remove duplicates
        unique_pts = []
        for p in pts:
            if not any(np.linalg.norm(p - u) < 1e-12 for u in unique_pts):
                unique_pts.append(p)
        pts = np.array(unique_pts)
        if len(pts) < 4:
            return np.pi
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    # For each facet (simplex), compute the circumradius
    # For a simplex on the unit sphere, the circumradius is the angular distance
    # from the sphere center to the circumcenter of the facet (in 3D, the circumcenter
    # of the facet triangle/vertex set)
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        # Get the points forming this facet
        facet_pts = pts[simplex]
        
        # Compute the circumcenter of this facet in 3D
        # For a triangle, the circumcenter is the point equidistant from all three vertices
        # We need the circumradius in terms of angular distance on the sphere
        
        if len(simplex) == 3:
            # Triangle facet
            a, b, c = facet_pts
            
            # Compute the circumcenter of the triangle in 3D
            # Using the formula: circumcenter = a + (b-a) x (c-a) cross product method
            # Actually, let's use a more robust approach:
            # The circumcenter O satisfies |O-a|^2 = |O-b|^2 = |O-c|^2
            # This gives linear equations in O
            
            # Set up the system: 2*(b-a)·O = |b|^2 - |a|^2, etc.
            # Since points are on unit sphere, |a|=|b|=|c|=1, so RHS = 0
            # So 2*(b-a)·O = 0, 2*(c-a)·O = 0
            # This means O is perpendicular to both (b-a) and (c-a), so O is along the normal
            # But wait, that's for the plane through the three points. The circumcenter lies
            # on the line through the centroid perpendicular to the plane.
            
            # Actually, for points on the unit sphere, the circumcenter of the triangle
            # (as a 3D point) is what we need. Let me compute it properly.
            
            # The circumcenter of triangle abc satisfies:
            # |O-a|^2 = |O-b|^2 => 2(b-a)·O = |b|^2 - |a|^2 = 0 (since on unit sphere)
            # |O-a|^2 = |O-c|^2 => 2(c-a)·O = 0
            # So O is in the null space of [2(b-a); 2(c-a)]
            # This means O is parallel to (b-a) x (c-a)
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            # O = t * n for some t
            # |t*n - a|^2 = |a|^2 - 2t*n·a + t^2 = 1 - 2t*n·a + t^2
            # We want this to equal |t*n - b|^2 = 1 - 2t*n·b + t^2
            # These are automatically equal since n·(b-a) = 0 (n is perpendicular to b-a)
            # Wait, that's not right. Let me reconsider.
            
            # Actually, the circumcenter in 3D for a triangle on the unit sphere:
            # The circumcenter is the center of the sphere passing through a, b, c.
            # Since a, b, c are on the unit sphere, the unit sphere itself passes through them.
            # But the circumcenter of the triangle (as a 2D object in its plane) is different.
            
            # Let me think again. The covering radius for a facet is the angular distance
            # from the farthest point on the sphere to the nearest configuration point.
            # For a Voronoi cell, the farthest point is the circumcenter of the facet
            # (the point equidistant from all vertices of the facet).
            
            # For a triangular facet with vertices a, b, c on the unit sphere, the 
            # circumcenter (in the sense of the point on the sphere equidistant from a, b, c)
            # is the point x on the unit sphere maximizing the minimum angular distance... no.
            
            # The circumcenter of the facet is the point in R^3 equidistant from a, b, c.
            # This point, when projected onto the unit sphere, gives the farthest point
            # in the Voronoi cell.
            
            # The circumcenter O of triangle abc satisfies |O-a| = |O-b| = |O-c|.
            # As derived: O is along the normal direction from the plane.
            # Let's find t such that O = t*n satisfies |O-a|^2 = |O-b|^2.
            # |t*n - a|^2 = t^2 - 2t(n·a) + 1
            # |t*n - b|^2 = t^2 - 2t(n·b) + 1
            # Equal when n·a = n·b, which is true since n is perpendicular to b-a.
            # So any t works? No, that can't be right.
            
            # I think I'm confusing things. Let me just compute the circumradius directly.
            # The circumradius R of a triangle with sides a, b, c is R = abc/(4*Area).
            # But we need the angular circumradius on the sphere.
            
            # Alternative approach: The covering radius is the maximum over all points y
            # on the sphere of min_i angular_distance(y, x_i).
            # This equals the maximum over all Voronoi vertices of their distance to the
            # nearest site. Voronoi vertices correspond to circumcenters of hull facets.
            
            # For a facet with vertices v1, v2, ..., vk, the circumcenter is the point
            # equidistant from all vi. The angular distance from this circumcenter to any
            # vertex is the circumradius.
            
            # Let me compute the circumcenter properly.
            # For points on the unit sphere, the circumcenter O satisfies:
            # |O - vi|^2 = R^2 for all i, where R is the Euclidean circumradius.
            # The angular circumradius is arccos(O·v1 / (|O| * |v1|)) = arccos(O·v1 / |O|)
            # since |v1| = 1.
            
            # To find O: O - vi is perpendicular to the plane... no.
            # O satisfies (O - vi) · (vj - vi) = 0 for j != i? No.
            # |O - vi|^2 = |O - vj|^2 => 2(vj - vi)·O = |vj|^2 - |vi|^2 = 0
            # So (vj - vi)·O = 0 for all j.
            # This means O is perpendicular to all (vj - vi), i.e., O is along the normal
            # to the plane containing the facet.
            
            # So O = t * n where n is the unit normal to the facet plane.
            # Now, |O - v1|^2 = |t*n - v1|^2 = t^2 - 2t(n·v1) + 1
            # This should be the same for all vertices, which it is since n·vi is the same
            # for all i (they're all in the same plane).
            
            # But we need another condition. The circumcenter is the center of the sphere
            # passing through all facet vertices. For a triangle, there's a unique such sphere
            # (unless degenerate). The center is determined by the three equations.
            
            # Wait, I think the issue is that for points on the unit sphere, the "circumcenter"
            # of the facet in the covering sense is actually the point on the line through
            # the origin and the plane normal that is equidistant from all vertices.
            
            # Let me just use a different approach: compute the covering radius by
            # finding the maximum of the distance to nearest point over a dense grid,
            # or by computing the Voronoi diagram on the sphere.
            
            # Actually, let's use a simpler numerical approach: for each facet, the
            # circumcenter is the point equidistant from all facet vertices. We can
            # solve for it directly.
            
            # For a triangle with vertices a, b, c on the unit sphere:
            # O = alpha*a + beta*b + gamma*c where the constraints are |O-a|=|O-b|=|O-c|
            # and O is in the plane of the triangle... no, O is not necessarily in the plane.
            
            # Let me just compute it numerically using the fact that O lies along the
            # normal from the centroid of the triangle.
            
            centroid = (a + b + c) / 3.0
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            # O = centroid + t * n
            # |O - a|^2 = |O - b|^2 is automatically satisfied.
            # We need |O - a|^2 to be minimized? No, the circumcenter is unique.
            
            # Actually, for three points, the circumcenter is uniquely determined.
            # Let me solve the linear system directly.
            # |O - a|^2 = |O - b|^2 => 2(b-a)·O = 0
            # |O - a|^2 = |O - c|^2 => 2(c-a)·O = 0
            # |O - a|^2 = R^2 (unknown)
            
            # From the first two: O is perpendicular to (b-a) and (c-a), so O = t*n.
            # Then R^2 = |t*n - a|^2 = t^2 - 2t(n·a) + 1
            # But we have 3 unknowns (t and R) and only 1 equation. We need another constraint.
            
            # I think the issue is that for 3 points, the circumcenter is not unique unless
            # we also require it to be in the plane or something. Actually, for 3 non-collinear
            # points, the circumcenter IS unique. Let me re-derive.
            
            # |O-a|^2 = |O-b|^2 => 2(b-a)·O = |b|^2 - |a|^2 = 0
            # |O-a|^2 = |O-c|^2 => 2(c-a)·O = |c|^2 - |a|^2 = 0
            # These two equations determine O up to a 1D family (since the two vectors b-a and
            # c-a span a 2D subspace, and O must be orthogonal to both, so O is along n).
            # So O = t*n, and t is free? That can't be right.
            
            # Oh wait, I think the circumcenter in 3D for 3 points is indeed a line, not a point.
            # The circumcenter is a point only when we have 4 points (a tetrahedron).
            # For 3 points, the set of points equidistant from all three is a line (the normal
            # to the plane through the centroid).
            
            # So for the covering radius, we need the point on this line that is on the unit
            # sphere? No, the circumcenter of the Voronoi cell is the point equidistant from
            # all facet vertices, and it's the point that maximizes the minimum distance.
            
            # I think the correct approach is: the covering radius for a facet is the angular
            # distance from the circumcenter of the facet (the point equidistant from all
            # facet vertices, chosen to be on the unit sphere) to any vertex.
            
            # For a triangle on the unit sphere, the circumcenter on the sphere is the point
            # x on the unit sphere such that x·a = x·b = x·c (equidistant in dot product).
            # This is the point in the intersection of the unit sphere and the null space
            # of [a-b; a-c], i.e., x is along n = (a-b) x (a-c).
            
            x = n / np.linalg.norm(n)
            # Check: x·a should equal x·b and x·c
            # Since x is perpendicular to a-b and a-c, x·a = x·b and x·a = x·c. Good.
            
            # The angular distance from x to a is arccos(x·a)
            angular_r = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
            max_circumradius = max(max_circumradius, angular_r)
            
        elif len(simplex) == 4:
            # Tetrahedron facet (shouldn't happen for convex hull in 3D, but just in case)
            # For a tetrahedron, the circumcenter is the point equidistant from all 4 vertices
            a, b, c, d = facet_pts
            # Solve for O such that |O-a|^2 = |O-b|^2 = |O-c|^2 = |O-d|^2
            # 2(b-a)·O = 0, 2(c-a)·O = 0, 2(d-a)·O = 0
            # O is in the null space of [b-a; c-a; d-a]
            # For 4 points in general position, the null space is {0}, so O = 0
            # But O=0 is the origin, which is inside the sphere. The circumradius would be
            # the angular distance from the origin to a vertex, which is 90 degrees.
            # This doesn't make sense for a covering radius.
            # I think for 4-point facets, we should handle them differently.
            # For now, skip.
            pass
        else:
            # Other cases
            pass
    
    return max_circumradius


def _cover_radius_fast(pts):
    """Faster covering radius computation using a grid-based approach."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Use a Fibonacci sphere grid for evaluation points
    n_grid = 5000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from grid points to configuration points
    # Using dot products for angular distance
    dots = grid @ pts.T  # shape (n_grid, N)
    min_dots = np.max(dots, axis=1)  # max dot product = min angular distance
    max_angular_dist = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    return np.max(max_angular_dist)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts)
    
    # Use gradient-based optimization to minimize the covering radius
    # We'll use a soft-max approach: minimize the log-sum-exp of distances
    
    def objective_and_grad(x):
        """Objective: soft-max of angular distances, gradient for projected gradient descent."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return np.inf, np.zeros_like(x)
        
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        # Compute covering radius using a moderate grid
        n_grid = 2000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, None))
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = grid @ p.T  # (n_grid, N)
        # Soft-max: log(sum(exp(beta * d))) where d is angular distance
        # Angular distance = arccos(dot), so we want to minimize max arccos(dot)
        # Equivalent to maximizing min dot, or minimizing -min dot
        # Use soft-min of dot products: -log(sum(exp(-beta * dot)))
        beta = 50.0
        neg_dots = -dots
        max_neg = np.max(neg_dots, axis=1, keepdims=True)
        soft_min = -np.log(np.sum(np.exp(neg_dots - max_neg), axis=1)) - max_neg.squeeze()
        # soft_min[i] ≈ -min_j dot(grid[i], p[j])
        # We want to minimize the max of -soft_min, i.e., maximize the min dot
        
        # Objective: sum of soft_min (we want to maximize min dot, so minimize -soft_min)
        obj = -np.mean(soft_min)
        
        # Gradient with respect to p
        # d(obj)/d(p) = -d(mean(soft_min))/d(p)
        # soft_min[i] = -log(sum_j exp(-beta * dot(grid[i], p[j]))) + max_j(-beta*dot(...))
        # This is getting complicated. Let's use a simpler gradient.
        
        # Alternative: use the gradient of the max of arccos(dot)
        # For the grid point with maximum angular distance, the gradient points
        # away from the nearest configuration point.
        
        min_dots_per_grid = np.max(dots, axis=1)
        max_idx = np.argmax(-min_dots_per_grid)  # grid point with worst coverage
        
        # Gradient: push the nearest point away from the worst grid point
        worst_grid = grid[max_idx]
        nearest_idx = np.argmax(dots[max_idx])
        
        grad = np.zeros_like(x)
        # Move p[nearest_idx] in the direction of worst_grid
        grad[nearest_idx * 3:(nearest_idx + 1) * 3] = worst_grid
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        return obj, grad
    
    # Use a simple projected gradient descent
    lr = 0.01
    for it in range(500):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        # Compute current radius
        current_radius = _cover_radius_fast(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Get gradient
        x_flat = pts.flatten()
        obj, grad = objective_and_grad(x_flat)
        
        if not np.isfinite(obj):
            break
        
        # Gradient step
        pts = pts - lr * grad.reshape(-1, 3)
        
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Adaptive learning rate
        new_radius = _cover_radius_fast(pts)
        if new_radius > current_radius:
            lr *= 0.8
            # Undo the step
            pts = pts + lr * grad.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
        else:
            lr = min(lr * 1.1, 0.05)
    
    # Final check
    final_radius = _cover_radius_fast(best_pts)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_pts
# EVOLVE-BLOCK-END
