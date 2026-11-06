"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130
DEADLINE = 125.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute the covering radius (in degrees) of points on the unit sphere."""
    # The covering radius is the largest angular distance from any point on the sphere
    # to its nearest site. This equals the maximum circumradius of the convex hull facets.
    # We approximate by checking the Voronoi vertices (circumcenters of Delaunay triangles).
    # For efficiency, we use the fact that the covering radius equals the maximum over
    # all pairs/triples of points of the circumradius of the spherical triangle.
    
    # A practical approach: the covering radius is the maximum over all facets of the
    # convex hull of the distance from the facet's circumcenter (projected to sphere) 
    # to the nearest site.
    
    # Simpler: use the fact that the covering radius equals the maximum angular distance
    # from any point on the sphere to its nearest site. We can estimate this by looking at
    # the circumcenters of the Delaunay triangulation on the sphere.
    
    # For a robust computation, we'll compute pairwise distances and find the maximum
    # "empty cap" radius. The covering radius is the maximum over all points y on the
    # sphere of min_i angle(y, p_i). The worst-case y is at a vertex of the Voronoi diagram,
    # which corresponds to the circumcenter of a Delaunay triangle.
    
    # Efficient approach: compute the convex hull and for each face compute the circumradius.
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max pairwise distance / 2 as rough upper bound
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, 0.0)
        d = np.maximum(d, 0.0)
        return np.degrees(np.arccos(np.clip(1.0 - np.max(d), -1.0, 1.0)))
    
    max_cos = 1.0  # cos of covering radius (we want min cos, i.e., max angle)
    
    for simplex in hull.simplices:
        # simplex is a set of 3 indices (for 3D convex hull, facets are triangles)
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of triangle abc projected onto the unit sphere.
        # The circumcenter in 3D of a triangle on the unit sphere lies along the normal
        # to the plane containing a, b, c.
        # Normal vector = (b-a) x (c-a)
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter is at distance d from origin along n, where d = 1/cos(theta)
        # and theta is the angular radius. Actually, the circumcenter of the spherical
        # triangle is the point on the sphere equidistant from a, b, c.
        
        # For points on the unit sphere, the circumcenter (on the sphere) is in the
        # direction of the normalized sum of the vertices if they form an acute triangle,
        # but more generally we need to find the point x on the sphere equidistant from a, b, c.
        
        # x · a = x · b = x · c = cos(r), |x| = 1
        # This means x is perpendicular to (a-b) and (a-c), so x is parallel to n.
        # x = ±n. We pick the sign such that x is on the same side as the triangle.
        # The circumradius r satisfies cos(r) = x · a.
        
        # Determine sign: x should be on the same side as the origin relative to the facet
        # (i.e., inside the convex hull). The normal n points outward. The circumcenter
        # of the spherical triangle is in the direction of -n (inward) if the triangle
        # is "small" (less than a hemisphere).
        
        # Actually, for the covering radius, we want the point on the sphere that is
        # equidistant from a, b, c and is the farthest from all sites. This is the
        # circumcenter of the spherical triangle. It lies along ±n.
        
        # Check both signs and pick the one that gives a valid circumcenter
        # (equidistant from a, b, c).
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            # Check if x is equidistant (within tolerance)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                # This is a circumcenter. The angular distance is arccos(da).
                # We want the maximum such distance over all facets.
                cos_r = da
                if cos_r < max_cos:
                    max_cos = cos_r
                break
    
    if max_cos >= 1.0:
        max_cos = 1.0 - 1e-10
    return np.degrees(np.arccos(np.clip(max_cos, -1.0, 1.0)))


def _soft_covering_radius(pts, beta=50.0):
    """Differentiable surrogate for the covering radius using a soft-max."""
    # The covering radius is max_y min_i angle(y, p_i).
    # We approximate by sampling points and using a soft-min.
    # For efficiency, we use the fact that the max occurs at Voronoi vertices.
    # As a smooth surrogate, we use: -log(sum_i exp(-beta * angle_i)) / beta
    # but this requires evaluating over a grid.
    
    # Instead, use the convex hull facet circumradii as before, but this isn't smooth.
    # For L-BFGS, we need a smooth objective. Let's use a different approach.
    
    # Compute the minimum distance from each facet circumcenter to its 3 defining points.
    # This is equivalent to the covering radius.
    
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 10.0  # large penalty
    
    max_angle = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                angle = np.arccos(np.clip(da, -1.0, 1.0))
                if angle > max_angle:
                    max_angle = angle
                break
    
    return np.degrees(max_angle)


def _optimize_from(pts, time_limit=30.0):
    """Optimize a starting configuration using L-BFGS-B on a smooth surrogate."""
    start_time = time.time()
    
    def _smooth_obj(flat_pts):
        """Smooth objective: maximize the minimum pairwise separation, then refine."""
        p = flat_pts.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / np.maximum(norms, 1e-12)
        
        # Compute pairwise dot products
        d = p @ p.T
        np.fill_diagonal(d, 1.0)
        
        # The objective we minimize: a surrogate for the covering radius.
        # We use the negative of the minimum over all pairs of (1 - dot),
        # which encourages spreading. But this doesn't directly minimize covering radius.
        
        # Better: use the sum of exp(-beta * min_distance_to_nearest) over a grid.
        # For speed, use a coarse grid.
        
        # Actually, let's use a simpler smooth surrogate:
        # minimize the maximum over i of (1 - min_{j != i} dot(p_i, p_j))
        # This encourages uniform spacing.
        
        # Even better: minimize the covering radius directly using the facet approach,
        # but it's not smooth. Let's use a penalty-based approach.
        
        # For L-BFGS, let's minimize a smooth function that correlates with covering radius:
        # f = sum_i exp(-lambda * min_{j != i} angular_distance(i, j))
        # This is the "energy" of the configuration, and minimizing it spreads points.
        
        # Compute minimum angular distance for each point
        min_dots = np.min(d, axis=1)  # min dot with any other point
        min_dots = np.clip(min_dots, -1.0, 1.0)
        min_angles = np.arccos(min_dots)
        
        # Soft-max of the angles (we want to minimize the max angle between neighbors)
        beta = 10.0
        max_angle = np.log(np.sum(np.exp(beta * min_angles))) / beta
        
        # Also add a term to encourage the points to be well-distributed
        # Use the covering radius estimate from facets as a secondary term
        return np.degrees(max_angle)
    
    # Start with the given points, flatten
    x0 = pts.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        _smooth_obj,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
    )
    
    # Get optimized points
    p_opt = result.x.reshape(-1, 3)
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    p_opt = p_opt / np.maximum(norms, 1e-12)
    
    return p_opt, time.time() - start_time


def _refine_covering(pts, time_limit=20.0):
    """Refine points to minimize the actual covering radius."""
    start_time = time.time()
    
    # Use the facet circumradius as objective, but make it smooth by
    # using a soft version.
    
    def _facet_obj(flat_pts):
        p = flat_pts.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / np.maximum(norms, 1e-12)
        return _soft_covering_radius(p)
    
    x0 = pts.flatten()
    result = minimize(
        _facet_obj,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
    )
    
    p_opt = result.x.reshape(-1, 3)
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    p_opt = p_opt / np.maximum(norms, 1e-12)
    
    return p_opt, time.time() - start_time


def construct():
    global DEADLINE
    t_start = time.time()
    
    # Step 1: Generate multiple diverse starting configurations
    rng = np.random.RandomState(42)
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Start 2: Random points
    pts_rand = rng.randn(N, 3)
    pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
    
    # Start 3: Another spiral with different phase
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + 1.0
    r2 = np.sqrt(1.0 - z2 * z2)
    pts_fib2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    
    starts = [pts_fib, pts_rand, pts_fib2]
    
    best_pts = None
    best_score = np.inf
    
    for s_idx, start_pts in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 40:
            break
        
        # Step 2: Optimize with L-BFGS
        try:
            opt_pts, t1 = _optimize_from(start_pts, time_limit=25.0)
        except Exception:
            continue
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 25:
            break
        
        # Step 3: Refine covering radius
        try:
            ref_pts, t2 = _refine_covering(opt_pts, time_limit=15.0)
        except Exception:
            ref_pts = opt_pts
        
        # Evaluate
        try:
            score = _covering_radius(ref_pts)
        except Exception:
            score = np.inf
        
        if score < best_score:
            best_score = score
            best_pts = ref_pts
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15:
            break
    
    # If we didn't find anything, use the Fibonacci spiral
    if best_pts is None:
        best_pts = pts_fib
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-12)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Perturb
                best_pts[j] += rng.randn(3) * 1e-4
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
