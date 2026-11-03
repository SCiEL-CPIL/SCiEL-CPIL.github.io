"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import scipy.optimize as opt

def _covering_radius(points):
    """Compute covering radius in radians using convex hull facets."""
    # Compute convex hull of points on sphere
    # The covering radius is the max over all points on sphere of min distance to a point
    # This equals the max circumradius of the spherical Delaunay triangles
    # For points on unit sphere, we can use the convex hull in 3D
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute force with many sample points
        theta = np.linspace(0, np.pi, 500)
        phi = np.linspace(0, 2*np.pi, 500)
        T, P = np.meshgrid(theta, phi)
        samples = np.column_stack([
            np.sin(T.ravel())*np.cos(P.ravel()),
            np.sin(T.ravel())*np.sin(P.ravel()),
            np.cos(T.ravel())
        ])
        # Compute min distance from each sample to each point
        # dot products
        dots = samples @ points.T  # (M, N)
        # angular distance = arccos(clamp(dot, -1, 1))
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # max dot = min angular distance
        max_ang = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_ang)
    
    # For each facet, compute the circumradius
    # A facet is a triangle on the convex hull
    # The circumcenter of the spherical triangle gives the farthest point in that cell
    max_radius = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Find the point on the sphere equidistant from v1, v2, v3 (circumcenter of spherical triangle)
        # This is the center of the empty cap
        # Solve: |x - v1|^2 = |x - v2|^2 = |x - v3|^2 with |x|=1
        # Equivalent to: x·v1 = x·v2 = x·v3
        # So x is perpendicular to (v1-v2) and (v1-v3)
        # x is parallel to cross(v1-v2, v1-v3) or similar
        a = v2 - v1
        b = v3 - v1
        # x is in the span of v1,v2,v3 and perpendicular to a and b in the plane
        # Actually, x·a = 0 and x·b = 0, so x is parallel to a×b
        x = np.cross(a, b)
        if np.linalg.norm(x) < 1e-12:
            # Degenerate
            continue
        x = x / np.linalg.norm(x)
        # Check both x and -x
        for xc in [x, -x]:
            # Check if xc is on the correct side (outside the hull, i.e., the empty cap center)
            # The circumcenter should be such that it's farther from the hull
            # Compute angular distances
            d1 = np.arccos(np.clip(np.dot(xc, v1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(xc, v2), -1, 1))
            d3 = np.arccos(np.clip(np.dot(xc, v3), -1, 1))
            r = max(d1, d2, d3)
            # Verify it's actually the farthest point: check that the cap centered at xc with radius r
            # doesn't contain any other points
            # For now, just take the max
            max_radius = max(max_radius, r)
    return max_radius

def _refine(points, max_time=100):
    """Iteratively refine points to minimize covering radius."""
    points = points.copy()
    start_time = time.time()
    
    # Use a repulsion/relaxation approach
    # For each iteration, compute the covering radius and push points
    # to reduce the largest empty cap
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Gradient-based refinement using soft-max
    def soft_covering_radius(pts, beta=50.0):
        """Smooth approximation of covering radius using log-sum-exp."""
        # Sample points on sphere
        # Use a fixed set of sample points for efficiency
        # Generate Fibonacci samples
        n_samples = 2000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Compute angular distances from each sample to each point
        dots = samples @ pts.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distance = arccos(dot), but we want min over points
        # min angular distance = arccos(max dot)
        max_dots = np.max(dots, axis=1)  # (M,)
        # Soft max of angular distances
        # ang = arccos(max_dots)
        # Use soft-max: (1/beta) * log(sum(exp(beta * ang)))
        # But arccos is expensive. Instead, minimize max_dots is equivalent to maximizing min distance
        # Actually, covering radius = max over samples of arccos(max_dots)
        # = arccos(min over samples of max_dots)
        # So minimizing covering radius = maximizing min over samples of max_dots
        # Let's use: f = -min(max_dots) and minimize f
        # Or use soft version: f = (1/beta)*log(sum(exp(-beta*max_dots)))
        
        # For gradient-based, let's directly work with max_dots
        # We want to maximize the minimum of max_dots
        # Use soft-min: (1/beta)*log(sum(exp(-beta*max_dots)))
        # and minimize this
        
        # For efficiency, use a smaller sample set in gradient computation
        return (1.0/beta) * np.log(np.sum(np.exp(-beta * max_dots)))
    
    # Iterative local search: for each point, try small perturbations
    # and accept if covering radius decreases
    
    n_iter = 0
    while time.time() - start_time < max_time:
        n_iter += 1
        # For each point, compute a gradient-like direction
        # Push points away from the centers of the largest empty caps
        
        # Compute convex hull and find the largest facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        best_facet = None
        best_r = 0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            a = v2 - v1
            b = v3 - v1
            x = np.cross(a, b)
            nx = np.linalg.norm(x)
            if nx < 1e-12:
                continue
            x = x / nx
            for xc in [x, -x]:
                d1 = np.arccos(np.clip(np.dot(xc, v1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(xc, v2), -1, 1))
                d3 = np.arccos(np.clip(np.dot(xc, v3), -1, 1))
                r = max(d1, d2, d3)
                if r > best_r:
                    best_r = r
                    best_facet = (simplex, xc, r)
        
        if best_facet is None:
            break
        
        simplex, xc, r = best_facet
        if r < 1e-6:
            break
        
        # Push the three points of the largest facet slightly away from xc
        # to reduce the empty cap size
        step = 0.01
        for idx in simplex:
            v = points[idx]
            # Push v in the direction away from xc
            # The direction to push is the component of -xc perpendicular to v
            # Actually, we want to move v so that the cap centered at xc shrinks
            # Moving v toward xc would shrink the cap, but v must stay on sphere
            # The optimal move is to move v in the direction of the projection of xc onto the tangent plane at v
            # tangent direction = xc - (xc·v)v, normalized
            dot_xc_v = np.dot(xc, v)
            tang = xc - dot_xc_v * v
            ntang = np.linalg.norm(tang)
            if ntang > 1e-10:
                tang = tang / ntang
                # Move v toward xc along the tangent
                new_v = v + step * tang
                new_v = new_v / np.linalg.norm(new_v)
                points[idx] = new_v
        
        # Renormalize
        for i in range(N):
            points[i] = points[i] / np.linalg.norm(points[i])
        
        # Check improvement
        new_radius = _covering_radius(points)
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = points.copy()
        else:
            # Revert if no improvement
            points = best_points.copy()
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine
    points = _refine(points, max_time=110)
    
    # Final check and normalize
    for i in range(N):
        points[i] = points[i] / np.linalg.norm(points[i])
    
    return points
# EVOLVE-BLOCK-END
