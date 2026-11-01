"""Improved covering of the sphere with 50 caps via repulsive-force relaxation."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of pts on the unit sphere.
    The covering radius is the maximum angular distance from any point on the
    sphere to the nearest site. This equals the maximum circumradius among the
    facets of the convex hull that have an empty cap (i.e., the facet's
    circumcenter is on the sphere side away from the origin, or equivalently,
    we take the facet whose normal points outward and compute the angle from
    the facet plane to the sphere).
    
    For a set of points on the unit sphere, the covering radius is the maximum
    over all facets of the convex hull of the angle between the facet normal
    (pointing outward from the origin) and any vertex of that facet. Actually,
    more precisely: for each facet, the circumcenter of that facet (projected
    to the sphere) gives a point on the sphere, and the angular distance from
    that point to the nearest site is what we want. But actually the covering
    radius equals the maximum, over all points y on the sphere, of min_i angle(y, x_i).
    
    A standard approach: the covering radius equals the maximum circumradius of
    the spherical Voronoi cells, which can be found by looking at the convex
    hull facets. For each facet with vertices v_1, ..., v_k, the circumcenter
    of the facet (in 3D) projected onto the sphere gives a candidate worst point.
    The angular distance from that point to the nearest site is the circumradius
    of that facet's spherical cell.
    
    More directly: for each facet, compute the normal n (pointing away from origin).
    The point on the sphere farthest from all sites within that facet's region is
    along n. The angular distance from n to the nearest site is arccos(max_i n·x_i).
    The covering radius is the max over facets of this quantity.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    covering_rad = 0.0
    for simplex in hull.simplices:
        face = pts[simplex]
        # Compute normal of the facet
        if len(face) < 3:
            continue
        # Normal via cross product of two edges
        edge1 = face[1] - face[0]
        edge2 = face[2] - face[0]
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        # Ensure normal points away from origin
        if np.dot(normal, face[0]) < 0:
            normal = -normal
        # Angular distance from normal to nearest site
        dots = np.dot(pts, normal)
        dots = np.clip(dots, -1.0, 1.0)
        max_dot = np.max(dots)
        angle = np.arccos(max_dot)
        if angle > covering_rad:
            covering_rad = angle
    return covering_rad


def _repulsive_force(pts, power=4):
    """
    Compute repulsive forces on each point. The force on point i is the sum over
    j != i of (x_i - x_j) / ||x_i - x_j||^power, projected tangent to the sphere.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                continue
            # Force magnitude ~ 1/dist^power, direction from j to i
            dist = np.sqrt(dist_sq)
            mag = 1.0 / (dist ** power)
            # Project diff to be tangent to sphere at pts[i]
            # Tangent projection: v - (v·n)n where n = pts[i]
            v = diff
            v_tangent = v - np.dot(v, pts[i]) * pts[i]
            forces[i] += mag * v_tangent
    return forces


def _relax(pts, max_iters=2000, lr_init=0.05, lr_decay=0.999, min_lr=1e-6, power=4):
    """
    Relax points on the sphere using repulsive forces with adaptive step size.
    """
    pts = _normalize(pts).copy()
    lr = lr_init
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        if lr < min_lr:
            break
        
        forces = _repulsive_force(pts, power=power)
        
        # Take a step
        new_pts = pts + lr * forces
        new_pts = _normalize(new_pts)
        
        # Check if this improves the covering radius (rough check every few iters)
        if it % 10 == 0:
            rad = _covering_radius(new_pts)
            if rad < best_rad:
                best_rad = rad
                best_pts = new_pts.copy()
        
        pts = new_pts
        lr *= lr_decay
    
    return best_pts, best_rad


def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_rad = _covering_radius(best_pts)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate initial points
        if restart == 0:
            pts = _fibonacci_sphere(N)
        else:
            pts = _random_sphere(N, rng)
        
        # Relax
        pts_relaxed, rad = _relax(pts, max_iters=1500, lr_init=0.05, lr_decay=0.999, power=4)
        
        if rad < best_rad:
            best_rad = rad
            best_pts = pts_relaxed.copy()
    
    # Final fine-tuning on the best solution
    if time.time() < _DEADLINE - 5:
        best_pts, best_rad = _relax(best_pts, max_iters=2000, lr_init=0.01, lr_decay=0.9995, power=6)
    
    return _normalize(best_pts)
