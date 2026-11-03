import numpy as np
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius for points on the unit sphere.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest point in 'points'.
    This is equal to the maximum circumradius of the spherical Voronoi cells.
    """
    try:
        vor = SphericalVoronoi(points)
        vertices = vor.vertices
        dots = vertices @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_dists = np.arccos(min_dots)
        return np.max(min_dists)
    except Exception:
        return np.pi / 2.0

def _compute_facet_circumradii_and_centers(points):
    """
    For each Delaunay triangle (facet of convex hull), compute:
    - the circumradius (angular distance from circumcenter to each vertex)
    - the circumcenter (point on sphere equidistant to the 3 vertices)
    Returns:
      - radii: array of shape (F,) with circumradius for each facet
      - centers: array of shape (F, 3) with circumcenters
      - simplices: array of shape (F, 3) with vertex indices
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        F = len(simplices)
        radii = np.zeros(F)
        centers = np.zeros((F, 3))
        
        for i, simp in enumerate(simplices):
            v1, v2, v3 = points[simp[0]], points[simp[1]], points[simp[2]]
            # Compute circumcenter of triangle on sphere
            # The circumcenter lies on the plane through origin and the normal to the triangle
            # Normal to triangle:
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-14:
                radii[i] = np.pi / 2.0
                centers[i] = v1
                continue
            n = n / norm_n
            
            # The circumcenter c satisfies:
            # dot(c, v1) = dot(c, v2) = dot(c, v3) = cos(r)
            # c is along n (or -n). Let's try c = alpha * n
            # dot(alpha * n, v1) = alpha * dot(n, v1) = cos(r)
            # dot(alpha * n, v2) = alpha * dot(n, v2) = cos(r)
            # dot(alpha * n, v3) = alpha * dot(n, v3) = cos(r)
            # So dot(n, v1) = dot(n, v2) = dot(n, v3) for a valid circumcenter along n.
            # This is true if n is the normal to the plane through v1,v2,v3 AND the plane passes through origin.
            # But the plane through v1,v2,v3 does NOT generally pass through origin.
            
            # Better approach: solve for c on sphere such that |c-v1|=|c-v2|=|c-v3|
            # |c-v1|^2 = |c|^2 + |v1|^2 - 2*c.dot(v1) = 2 - 2*c.dot(v1)
            # So c.dot(v1) = c.dot(v2) = c.dot(v3)
            # Let A = [v1; v2; v3], we want c such that A @ c = [k, k, k] for some k
            # And |c| = 1.
            # From A @ c = k * ones(3), we have (A - ones(3) @ mean_row) @ c = 0
            # Simpler: c is in the null space of [v1-v2; v2-v3]
            # v1-v2 and v2-v3 span a 2D subspace; c is orthogonal to both.
            # c is parallel to cross(v1-v2, v2-v3) = cross(v1-v2, v3-v2)
            d1 = v1 - v2
            d2 = v3 - v2
            c_dir = np.cross(d1, d2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                radii[i] = np.pi / 2.0
                centers[i] = v1
                continue
            c_dir = c_dir / c_norm
            
            # Check both directions
            # For c = c_dir: dot(c_dir, v1), dot(c_dir, v2), dot(c_dir, v3)
            # For c = -c_dir: dot(-c_dir, v1), etc.
            # The correct one has all three dots equal (or as close as possible)
            dots_pos = np.array([np.dot(c_dir, v1), np.dot(c_dir, v2), np.dot(c_dir, v3)])
            dots_neg = -dots_pos
            
            # Choose the direction where the three dots are more equal
            var_pos = np.var(dots_pos)
            var_neg = np.var(dots_neg)
            if var_neg < var_pos:
                c = -c_dir
                dots = dots_neg
            else:
                c = c_dir
                dots = dots_pos
            
            # The circumradius is arccos(mean(dots)) approximately
            # More precisely, for exact equidistant points, all dots are equal
            # Use the mean as a robust estimate
            cos_r = np.mean(dots)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radii[i] = np.arccos(cos_r)
            centers[i] = c
        
        return radii, centers, simplices
    except Exception:
        return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _compute_covering_radius(points)

    # 2. Lloyd's Algorithm (Spherical K-Means)
    for k in range(30):
        if time.time() > deadline - 40.0:
            break

        try:
            vor = SphericalVoronoi(points)
            new_points = np.zeros_like(points)
            for i in range(N):
                region = vor.regions[i]
                if not region:
                    new_points[i] = points[i]
                    continue
                verts = vor.vertices[region]
                c = np.sum(verts, axis=0)
                norm = np.linalg.norm(c)
                if norm < 1e-10:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / norm

            diff = np.linalg.norm(new_points - points, axis=1)
            if np.max(diff) < 1e-6:
                break

            points = new_points
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()

        except Exception:
            break

    # 3. Balancing force: equalize facet circumradii
    # For each point, compute the average circumradius of adjacent facets.
    # If a facet has a large circumradius, its circumcenter is far from the points.
    # Moving a point toward the circumcenter of a large-circumradius facet
    # reduces that facet's circumradius.
    # The "force" on point i is the weighted sum of (center_f - point_i)
    # for all facets f containing point i, weighted by how much larger
    # facet f's circumradius is compared to the mean.
    for it in range(200):
        if time.time() > deadline - 20.0:
            break

        try:
            radii, centers, simplices = _compute_facet_circumradii_and_centers(points)
            if radii is None:
                break

            F = len(radii)
            mean_r = np.mean(radii)
            # Weight: how much above average each facet's circumradius is
            # Clip to avoid negative weights (facets smaller than average don't push)
            weights = np.maximum(radii - mean_r, 0.0)
            # Normalize weights
            w_sum = np.sum(weights)
            if w_sum < 1e-15:
                # All circumradii are equal (or nearly); stop
                break
            weights = weights / w_sum

            # Compute force on each point
            forces = np.zeros_like(points)
            for f in range(F):
                w = weights[f]
                if w < 1e-15:
                    continue
                c = centers[f]
                for idx in simplices[f]:
                    # Direction from point to circumcenter
                    d = c - points[idx]
                    # Project to tangent plane (remove radial component)
                    d_tangent = d - np.dot(d, points[idx]) * points[idx]
                    forces[idx] += w * d_tangent

            # Normalize forces and take a step
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            unit_forces = forces / force_norms

            # Adaptive step size: start small, increase if improving
            step_size = 0.01
            new_points = points + step_size * unit_forces
            # Normalize to sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            new_points = new_points / norms

            # Evaluate
            new_radius = _compute_covering_radius(new_points)

            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                # Try to increase step size
                step_size = min(step_size * 1.5, 0.1)
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break

        except Exception:
            break

    # 4. Polish with Nelder-Mead on the covering radius
    def objective(flat_points):
        if time.time() > deadline:
            return 10.0
        pts = flat_points.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return _compute_covering_radius(pts)

    initial_flat = points.flatten()

    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})

        if res.fun < best_radius:
            best_radius = res.fun
            best_flat = res.x
            norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            best_points = best_flat.reshape(N, 3) / norms
    except Exception:
        pass

    # Final check and return
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms

    return best_points
