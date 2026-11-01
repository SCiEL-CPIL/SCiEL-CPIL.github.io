"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (max angular distance to nearest point) in radians.
    Uses the convex hull facets: for each facet, the circumradius of the facet's
    points gives the angular distance from the facet's pole to the nearest point.
    The covering radius is the max over all facets of this circumradius."""
    points = np.asarray(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each facet (simplex), compute the circumradius in angular terms
    # The circumradius of a spherical triangle with vertices a, b, c (unit vectors)
    # can be found by: the pole of the facet is n = a × b / |a × b| (for triangle),
    # and the angular distance from n to a is the circumradius.
    # More generally, for a facet with vertices v_0, ..., v_{k-1}, the circumradius
    # is the angular distance from the facet's pole to any vertex.
    
    max_rad = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]  # shape (k, 3)
        # Compute the pole (normal) of the facet
        # For a simplex, the pole is the normalized cross product of two edges
        # For a triangle: n = (v1 - v0) × (v2 - v0)
        # For a quadrilateral or higher, we can use the first 3 vertices to get an approximate pole
        # Actually, for the convex hull of points on the sphere, each facet is a planar polygon.
        # The pole is the direction from origin to the centroid of the facet (for a planar polygon
        # containing the origin in its "shadow"), but more accurately, the pole is the unit vector
        # perpendicular to the facet's plane pointing outward.
        
        # For a simplex (triangle), compute pole via cross product
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            # Angular distance from pole to any vertex
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_rad = max(max_rad, theta)
        else:
            # For higher-dimensional facets (shouldn't happen in 3D convex hull of points on sphere,
            # but just in case), use the first 3 vertices
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_rad = max(max_rad, theta)
    
    return max_rad


def _repulsive_forces(points, k=2.0, strength=1.0):
    """Compute repulsive forces on each point. Force on point i from point j is
    proportional to strength / d^k in the direction from j to i, where d is angular distance."""
    n = len(points)
    forces = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Angular distance
            cos_d = np.clip(np.dot(points[i], points[j]), -1.0, 1.0)
            d = np.arccos(cos_d)
            if d < 1e-10:
                # Very close, use random perturbation
                forces[i] += strength * (np.random.randn(3) * 0.01)
                continue
            # Force magnitude: strength / d^k
            f_mag = strength / (d ** k)
            # Direction: from j to i (projected onto tangent plane at i)
            direction = points[i] - np.dot(points[i], points[j]) * points[j]
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                continue
            direction = direction / dir_norm
            forces[i] += f_mag * direction
    return forces


def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with a random configuration, but avoid coincidences
    points = rng.standard_normal((N, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Run repulsive-force optimization
    n_iter = 500
    base_step = 0.05
    
    for it in range(n_iter):
        # Cooling schedule
        step = base_step * (1.0 - it / n_iter) + 0.001
        
        # Compute forces
        forces = _repulsive_forces(points, k=2.0, strength=1.0)
        
        # Update points
        points = points + step * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    # Final refinement: use Nelder-Mead on the covering radius
    def objective(x):
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _covering_radius(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
    
    points = result.x.reshape(N, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
