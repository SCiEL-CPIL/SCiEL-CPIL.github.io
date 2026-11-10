"""Covering the sphere with 130 equal caps – SA-based optimizer."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) via convex hull.
    The covering radius is the max over all facets of the circumradius
    of that facet (the angle from the facet center to the farthest vertex).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: very large

    # For each facet, compute the circumradius
    # Facet vertices are given by hull.simplices
    max_circumradius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # Centroid of the facet
        center = verts.mean(axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-12:
            # Degenerate: use max angular distance to any vertex
            for v in verts:
                ang = np.arccos(np.clip(np.dot(center, v), -1.0, 1.0))
                if ang > max_circumradius:
                    max_circumradius = ang
            continue
        center = center / center_norm
        # The circumradius is the maximum angle from center to any vertex
        # Actually for a spherical cap covering, the relevant radius is the
        # angle from the facet's circumcenter to its vertices.
        # For a triangle on the sphere, the circumradius can be computed.
        # A simpler approach: the covering radius contribution from this facet
        # is the maximum angular distance from the facet's "center" (in the
        # spherical sense) to its vertices.
        # The spherical circumcenter of a triangle is the point equidistant
        # from all three vertices. For small triangles, the Euclidean centroid
        # projected to the sphere is a good approximation.
        # More accurately: the circumradius of the spherical triangle.
        # For a triangle with vertices a, b, c on the unit sphere, the
        # circumradius R satisfies: the angle from circumcenter to each vertex.
        # We can compute it as: for each pair, the angular distance, then
        # use the spherical law of cosines.
        # Simpler: the covering radius for this facet is the max angular
        # distance from the facet center (spherical circumcenter) to vertices.
        # Approximation: use the Euclidean circumcenter of the triangle projected
        # to the sphere.
        a, b, c = verts[0], verts[1], verts[2]
        # Compute circumcenter in 3D (Euclidean circumcenter of the triangle)
        # Then project to unit sphere.
        d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) -
                   b[0] * (a[1] * c[2] - c[1] * a[2]) +
                   c[0] * (a[1] * b[2] - b[1] * a[2]))
        if abs(d) < 1e-12:
            # Degenerate triangle
            for v in verts:
                ang = np.arccos(np.clip(np.dot(center, v), -1.0, 1.0))
                if ang > max_circumradius:
                    max_circumradius = ang
            continue
        ux = ((b[1]**2 + b[2]**2) * (a[1]*c[2] - c[1]*a[2]) -
              (a[1]**2 + a[2]**2) * (b[1]*c[2] - c[1]*b[2])) / d
        uy = ((a[0]**2 + a[2]**2) * (b[1]*c[2] - c[1]*b[2]) -
              (b[0]**2 + b[2]**2) * (a[1]*c[2] - c[1]*a[2])) / d
        uz = ((b[0]**2 + b[1]**2) * (a[0]*c[1] - c[0]*a[1]) -
              (a[0]**2 + a[1]**2) * (b[0]*c[1] - c[0]*b[1])) / d
        circumcenter = np.array([ux, uy, uz])
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-12:
            cc = center
        else:
            cc = circumcenter / cc_norm
        # Angular distance from circumcenter to each vertex
        for v in verts:
            cosang = np.clip(np.dot(cc, v), -1.0, 1.0)
            ang = np.arccos(cosang)
            if ang > max_circumradius:
                max_circumradius = ang
    return max_circumradius


def _covering_radius_fast(pts):
    """
    Faster covering radius: use the fact that the covering radius is the
    maximum over all points on the sphere of the distance to the nearest
    point. This equals the maximum circumradius of the Voronoi cells on
    the sphere, which for the convex hull corresponds to the facets.
    
    Alternative faster method: for each pair of points, check if the
    midpoint (normalized) is far from all points. But that's O(N^2 * N).
    
    Let's use the convex hull approach but vectorize where possible.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_cr = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        # Spherical circumradius: angle from circumcenter to vertices
        d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) -
                   b[0] * (a[1] * c[2] - c[1] * a[2]) +
                   c[0] * (a[1] * b[2] - b[1] * a[2]))
        if abs(d) < 1e-12:
            center = verts.mean(axis=0)
            cn = np.linalg.norm(center)
            if cn > 1e-12:
                center = center / cn
            else:
                continue
            for v in verts:
                ang = np.arccos(np.clip(np.dot(center, v), -1.0, 1.0))
                if ang > max_cr:
                    max_cr = ang
            continue
        ux = ((b[1]**2 + b[2]**2) * (a[1]*c[2] - c[1]*a[2]) -
              (a[1]**2 + a[2]**2) * (b[1]*c[2] - c[1]*b[2])) / d
        uy = ((a[0]**2 + a[2]**2) * (b[1]*c[2] - c[1]*b[2]) -
              (b[0]**2 + b[2]**2) * (a[1]*c[2] - c[1]*a[2])) / d
        uz = ((b[0]**2 + b[1]**2) * (a[0]*c[1] - c[0]*a[1]) -
              (a[0]**2 + a[1]**2) * (b[0]*c[1] - c[0]*b[1])) / d
        cc = np.array([ux, uy, uz])
        ccn = np.linalg.norm(cc)
        if ccn < 1e-12:
            continue
        cc = cc / ccn
        # Max angle to vertices
        dots = np.clip(cc @ verts.T, -1.0, 1.0)
        angs = np.arccos(dots)
        m = angs.max()
        if m > max_cr:
            max_cr = m
    return max_cr


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # Initialize with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_cr = _covering_radius_fast(pts)

    # Simulated Annealing
    T = 0.05  # initial temperature (radians)
    T_min = 0.0001
    cooling_rate = 0.9995

    np.random.seed(42)

    while T > T_min:
        if time.time() > deadline:
            break

        # Pick a random point and perturb it
        idx = np.random.randint(N)
        orig = pts[idx].copy()

        # Random perturbation: add Gaussian noise and renormalize
        noise = np.random.randn(3) * T
        new_pt = orig + noise
        norm = np.linalg.norm(new_pt)
        if norm < 1e-10:
            new_pt = orig
        else:
            new_pt = new_pt / norm

        pts[idx] = new_pt
        new_cr = _covering_radius_fast(pts)
        delta = new_cr - best_cr

        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
        else:
            pts[idx] = orig

        T *= cooling_rate

    # Final local refinement: try small moves on the best configuration
    pts = best_pts.copy()
    T_local = 0.001
    for _ in range(2000):
        if time.time() > deadline:
            break
        idx = np.random.randint(N)
        orig = pts[idx].copy()
        noise = np.random.randn(3) * T_local
        new_pt = orig + noise
        norm = np.linalg.norm(new_pt)
        if norm < 1e-10:
            continue
        new_pt = new_pt / norm
        pts[idx] = new_pt
        new_cr = _covering_radius_fast(pts)
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = pts.copy()
        else:
            pts[idx] = orig

    return _normalize(best_pts)
