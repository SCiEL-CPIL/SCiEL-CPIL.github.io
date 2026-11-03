"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius using convex hull facet circumradii.
    The covering radius is the maximum over all facets of the circumradius
    of that facet (the angular radius of the smallest cap containing the facet).
    For a triangle on the unit sphere, this is the angle from the centroid
    to the farthest vertex, but more precisely it's related to the circumcircle.
    
    Actually, the covering radius is the maximum distance from any point on the sphere
    to its nearest configuration point. This equals the maximum circumradius of the
    spherical triangles (facets) of the convex hull.
    
    For a spherical triangle with vertices a, b, c on the unit sphere, the circumradius
    R satisfies: the circumcenter is the point equidistant from all three vertices.
    For small triangles, R ≈ (abc)/(4*Area) where abc is the side length product.
    
    More precisely, for a triangle with side lengths a, b, c (angular distances),
    the circumradius R satisfies sin(R) = (sin(a)*sin(b)*sin(c)) / (4 * sin(s) * cos(s-a) * cos(s-b) * cos(s-c))
    where s = (a+b+c)/2.
    
    But a simpler approach: the covering radius is the maximum over all points y on the sphere
    of min_i angle(y, x_i). This is achieved at the circumcenter of some facet.
    
    Let me use a direct approach: for each facet (triangle) of the convex hull,
    compute the circumradius of that spherical triangle.
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use sampled approach
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 200)
        theta, phi = np.meshgrid(theta, phi)
        y = np.column_stack([
            np.sin(theta)*np.cos(phi).ravel(),
            np.sin(theta)*np.sin(phi).ravel(),
            np.cos(theta).ravel()
        ])
        # Compute min distance from each y to any x_i
        dots = y @ pts.T  # shape (n_samples, N)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angles = angles.min(axis=1)
        return np.max(min_angles)
    
    # For each facet (simplex), compute the circumradius
    # In 3D, facets are triangles
    max_radius = 0.0
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumradius of spherical triangle
            # Side lengths
            ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            # Use formula for circumradius of spherical triangle
            s = (ab + bc + ca) / 2.0
            sin_a = np.sin(ab)
            sin_b = np.sin(bc)
            sin_c = np.sin(ca)
            sin_s = np.sin(s)
            cos_sa = np.cos(s - ab)
            cos_sb = np.cos(s - bc)
            cos_sc = np.cos(s - ca)
            
            denom = 4.0 * sin_s * cos_sa * cos_sb * cos_sc
            if abs(denom) < 1e-15:
                # Degenerate case
                R = max(ab, bc, ca) / 2.0
            else:
                sin_R = (sin_a * sin_b * sin_c) / denom
                sin_R = np.clip(sin_R, -1.0, 1.0)
                R = np.arcsin(sin_R)
            
            max_radius = max(max_radius, R)
    
    return max_radius


def _fibonacci_points(n, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)


def _icosahedron_based_points(n, seed):
    """Generate points based on icosahedron vertices with some perturbation."""
    rng = np.random.RandomState(seed)
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Start with icosahedron vertices and add more points
    pts = [verts]
    while len(pts) < n:
        # Add random points
        new_pts = rng.randn(1, 3)
        new_pts = _normalize(new_pts)
        pts.append(new_pts)
    
    pts = np.vstack(pts[:n])
    return _normalize(pts)


def _optimize_from_start(start_pts, deadline, max_iter=1000):
    """Optimize starting from given points using Nelder-Mead on the actual objective."""
    start_time = time.time()
    best_pts = start_pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Flatten to 1D for optimization
    x0 = best_pts.ravel()
    
    def objective(x):
        if time.time() > deadline:
            return best_cr
        pts = _normalize(x.reshape(N, 3))
        return _covering_radius(pts)
    
    # Use Nelder-Mead
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        if result.success or result.fun < best_cr:
            best_pts = _normalize(result.x.reshape(N, 3))
            best_cr = result.fun
    except Exception:
        pass
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points(N, seed=0))
    
    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_random_points(N, seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points(N, seed=10))
    
    best_pts = initializations[0]
    best_cr = _covering_radius(best_pts)
    
    # Optimize from each start
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        # Check if this start is already better
        cr_init = _covering_radius(init_pts)
        if cr_init < best_cr:
            best_pts = init_pts.copy()
            best_cr = cr_init
        
        # Try to optimize
        try:
            opt_pts, opt_cr = _optimize_from_start(init_pts, deadline, max_iter=500)
            if opt_cr < best_cr:
                best_pts = opt_pts
                best_cr = opt_cr
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time allows
    if time.time() < deadline - 10.0:
        try:
            x0 = best_pts.ravel()
            
            def objective(x):
                if time.time() > deadline:
                    return best_cr
                pts = _normalize(x.reshape(N, 3))
                return _covering_radius(pts)
            
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_cr:
                best_pts = _normalize(result.x.reshape(N, 3))
                best_cr = result.fun
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
