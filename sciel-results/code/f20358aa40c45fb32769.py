"""Optimize 50-point sphere covering via projected gradient descent on the sphere."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # fallback: 90 degrees

    # For each facet, compute the angular distance from the facet's circumcenter to the facet vertices
    # The covering radius is the max over facets of the angular distance from the origin to the farthest point on the facet
    # which equals the angular radius of the circumcircle of the facet (if the circumcenter is on the same side)

    # A more robust approach: for each triangular facet, compute the circumcenter in 3D,
    # project to sphere, and compute the angular distance to each vertex.
    # The covering radius is the maximum of these over all facets.

    best = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        # Compute circumcenter of triangle on sphere
        # The circumcenter is the point equidistant from all three vertices
        # In 3D, the circumcenter of the triangle (as a planar triangle) projected to the sphere
        # Actually, we want the point on the sphere that is the center of the smallest cap containing the triangle.
        # This is the circumcenter of the spherical triangle.

        # Method: find the point c on the sphere that minimizes max(|c-v0|, |c-v1|, |c-v2|)
        # This is the center of the minimum enclosing circle of the three points on the sphere.

        # For a spherical triangle, the circumcenter can be found by:
        # 1. Compute the normal to the plane of the triangle
        # 2. The circumcenter in 3D space (not on sphere) is where the perpendicular bisectors meet
        # 3. Project to the sphere

        # Let's use a different approach: the covering radius for a facet is the angular distance
        # from the facet's "center" (the point on the sphere farthest from all three vertices in the cap)
        # Actually, the covering radius contribution from a facet is the maximum angular distance
        # from any point in the facet's cap to the nearest vertex.

        # Simpler: the circumradius of the spherical triangle.
        # The circumcenter of a spherical triangle is the intersection of the great circles
        # that are the perpendicular bisectors of the sides.

        # Let's compute it numerically: find c on unit sphere maximizing min angular distance to v0, v1, v2
        # This is equivalent to finding the point that minimizes the maximum distance.

        # Use the fact that for three points on a sphere, the circumcenter (if the triangle is not obtuse)
        # is the normalized cross product approach or we can solve it directly.

        # Direct approach: the circumcenter c satisfies:
        # |c - v0|^2 = |c - v1|^2 = |c - v2|^2
        # |c|^2 = 1
        # This gives: c·(v0-v1) = 0, c·(v0-v2) = 0, |c|^2 = 1
        # So c is parallel to (v0-v1) × (v0-v2) ... no that's the normal to the plane.

        # Actually: c·v0 = c·v1 = c·v2 (since |c-vi|^2 = 1 + 1 - 2 c·vi, so equal distances means equal dot products)
        # So c·(v0-v1) = 0 and c·(v0-v2) = 0
        # The solution is c = normalize((v0-v1) × (v0-v2)) ... wait no.
        # c is orthogonal to (v0-v1) and (v0-v2), so c is parallel to (v0-v1) × (v0-v2).

        e1 = v0 - v1
        e2 = v0 - v2
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            # Degenerate
            best = max(best, np.arccos(np.clip(np.dot(v0, v1), -1, 1)))
            continue
        c = normal / norm

        # Check which side: the circumcenter should be on the same side as the origin relative to the facet
        # Actually for covering, we want the point on the sphere that is farthest from all three vertices
        # within the facet's cap. The circumcenter candidate is c or -c.
        # We need the one that is "inside" the cap, i.e., the one where the angular distance to each vertex
        # is <= pi/2 typically, or more precisely, the one that maximizes the minimum distance to vertices
        # among the two candidates c and -c... actually we want the point that is the center of the largest
        # empty cap, which is the circumcenter.

        # Let's compute angular distance from c to each vertex
        d0 = np.arccos(np.clip(np.dot(c, v0), -1, 1))
        d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))

        # Also try -c
        d0b = np.arccos(np.clip(np.dot(-c, v0), -1, 1))
        d1b = np.arccos(np.clip(np.dot(-c, v1), -1, 1))
        d2b = np.arccos(np.clip(np.dot(-c, v2), -1, 1))

        # The covering radius for this facet is the max of the three distances for the correct center
        # The correct center is the one that is "outside" the triangle (farther from the centroid of the triangle)
        # Actually, for the covering problem, we want the point on the sphere that is the farthest from all
        # three vertices. This is the circumcenter if the triangle is acute, otherwise it's the longest edge's midpoint.

        # Let's just take the max of the min-distance approach:
        # The circumradius is max(d0, d1, d2) if c is the right one, or max(d0b, d1b, d2b)
        # We pick the candidate that gives the smaller max (the "inner" circumcenter)
        # Actually no - for covering radius, we want the LARGEST circumradius over all facets.
        # For each facet, the circumradius is determined by the geometry.

        # Let's just compute both and take the one that makes sense (distances should be <= pi)
        r1 = max(d0, d1, d2)
        r2 = max(d0b, d1b, d2b)

        # The circumcenter for the covering problem is the one that is NOT the antipode of the triangle's center
        # Typically, the circumcenter c should satisfy that the triangle is contained in the cap centered at c
        # with radius r1. Let's check: if all d0,d1,d2 < pi/2, then c is likely the right one.
        # Otherwise, -c might be.

        # Heuristic: pick the candidate with smaller max distance
        # But actually both are valid circumcenters (of the two possible circles through 3 points on sphere)
        # The one relevant for covering is the one that defines the facet's cap.

        # The facet's cap is the set of points closer to the facet than to any other facet.
        # The circumcenter of the facet is the point on the sphere equidistant from the three vertices.
        # For the convex hull, the relevant circumcenter is the one on the same side of the facet plane as the origin...
        # Actually, the origin is inside the convex hull, so the facet's outward normal points away from origin.
        # The circumcenter for the covering radius should be on the "outside" of the facet.

        # Let's use: the facet normal points outward. The circumcenter should be in the direction of the normal.
        # The outward normal of the facet is the normalized cross product in the correct order.
        # Hull simplices are ordered so that the normal points outward.

        # For scipy ConvexHull, the normal points outward.
        # Our c = normalize((v0-v1) x (v0-v2)) - we need to check if this points outward.
        # The centroid of the triangle is (v0+v1+v2)/3
        # The outward normal should point away from the origin (since origin is inside hull)
        # So dot(normal, centroid) > 0 for outward normal? No, dot(normal, centroid) should be > 0 if normal points outward.

        centroid = (v0 + v1 + v2) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            d0, d1, d2 = d0b, d1b, d2b
            r1 = max(d0, d1, d2)

        # The circumradius for this facet
        facet_radius = max(d0, d1, d2)
        best = max(best, facet_radius)

    return best * 180.0 / np.pi


def _compute_covering_radius_fast(pts):
    """Faster covering radius computation."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    best = 0.0
    simplices = hull.simplices
    for simplex in simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        e1 = v0 - v1
        e2 = v0 - v2
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        c = normal / norm

        centroid = (v0 + v1 + v2) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c

        d0 = np.arccos(np.clip(np.dot(c, v0), -1, 1))
        d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))

        facet_radius = max(d0, d1, d2)
        if facet_radius > best:
            best = facet_radius

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin

    # Initial configuration: golden spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms

    best_pts = pts.copy()
    best_radius = _compute_covering_radius_fast(pts)

    # Gradient descent on the sphere
    # The objective is the covering radius, which is nonsmooth.
    # We'll use a soft approximation: for each facet, compute the circumradius,
    # and use a soft-max to get a smooth objective.

    # Alternative: use the fact that the covering radius is determined by the "critical" facets.
    # We'll do iterative refinement:
    # 1. Compute covering radius
    # 2. Identify the facet(s) that achieve the maximum
    # 3. Move the vertices of those facets to reduce the circumradius

    # Let's use a simpler approach: projected gradient descent with a smooth surrogate.
    # Surrogate: sum of squared circumradii, or use log-sum-exp of circumradii.

    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break

        # Compute covering radius and identify critical facets
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Perturb points
            pts += np.random.randn(*pts.shape) * 0.01
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            continue

        simplices = hull.simplices
        facet_radii = []
        facet_centers = []

        for simplex in simplices:
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            e1 = v0 - v1
            e2 = v0 - v2
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                facet_radii.append(0.0)
                facet_centers.append(np.zeros(3))
                continue
            c = normal / norm
            centroid = (v0 + v1 + v2) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c

            d0 = np.arccos(np.clip(np.dot(c, v0), -1, 1))
            d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
            facet_radii.append(max(d0, d1, d2))
            facet_centers.append(c)

        facet_radii = np.array(facet_radii)
        facet_centers = np.array(facet_centers)

        # Soft-max weighting
        beta = 50.0
        shifted = facet_radii - facet_radii.max()
        weights = np.exp(beta * shifted)
        weights /= weights.sum()

        # Gradient: for each vertex, compute the weighted sum of gradients from facets containing it
        grad = np.zeros_like(pts)
        for idx, simplex in enumerate(simplices):
            w = weights[idx]
            if w < 1e-10:
                continue
            c = facet_centers[idx]
            r_i = facet_radii[idx]
            # The gradient of the circumradius with respect to each vertex
            # For a spherical triangle, the circumcenter c satisfies c·vi = constant
            # The circumradius r = arccos(c·vi)
            # dr/dvi = -c / sqrt(1 - (c·vi)^2) ... this is getting complex
            # Simpler: just move each vertex slightly away from the circumcenter if it's a critical facet
            for v_idx in simplex:
                vi = pts[v_idx]
                # Move vi away from c to increase the distance, which... wait
                # We want to DECREASE the circumradius.
                # The circumradius is the max distance from c to the vertices.
                # To decrease it, we should move the vertices closer to c.
                # But we also need to keep them on the sphere.
                # Gradient of |vi - c|^2 w.r.t. vi is 2(vi - c)
                # To minimize, we move vi in the direction -(vi - c) = c - vi
                # Projected to the sphere: vi_new = normalize(vi + lr * (c - vi))
                # But this is for a single facet. With multiple facets, we average.
                grad[v_idx] += w * (c - vi)

        # Update: move in the direction of the gradient (to reduce circumradius)
        # Actually, we want to minimize the circumradius, so we move vertices toward the circumcenters
        # of the critical facets.
        step = lr * grad
        # Project to sphere: pts_new = normalize(pts + step)
        pts_new = pts + step
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms

        # Check if the new configuration is valid
        new_radius = _compute_covering_radius_fast(pts_new)

        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = pts_new.copy()
            pts = pts_new
            # Decrease learning rate slowly
            lr *= 0.999
        else:
            # Reject and decrease learning rate
            lr *= 0.9
            if lr < 1e-5:
                break

    # Final polish: a few more iterations with smaller step
    if time.time() < deadline:
        lr = 0.001
        for _ in range(500):
            if time.time() > deadline:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            simplices = hull.simplices
            facet_radii = []
            facet_centers = []
            for simplex in simplices:
                v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                e1 = v0 - v1
                e2 = v0 - v2
                normal = np.cross(e1, e2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    facet_radii.append(0.0)
                    facet_centers.append(np.zeros(3))
                    continue
                c = normal / norm
                centroid = (v0 + v1 + v2) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                d0 = np.arccos(np.clip(np.dot(c, v0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
                facet_radii.append(max(d0, d1, d2))
                facet_centers.append(c)

            facet_radii = np.array(facet_radii)
            facet_centers = np.array(facet_centers)
            beta = 50.0
            shifted = facet_radii - facet_radii.max()
            weights = np.exp(beta * shifted)
            weights /= weights.sum()

            grad = np.zeros_like(pts)
            for idx, simplex in enumerate(simplices):
                w = weights[idx]
                if w < 1e-10:
                    continue
                c = facet_centers[idx]
                for v_idx in simplex:
                    vi = pts[v_idx]
                    grad[v_idx] += w * (c - vi)

            step = lr * grad
            pts_new = pts + step
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            new_radius = _compute_covering_radius_fast(pts_new)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = pts_new.copy()
                pts = pts_new
            else:
                lr *= 0.9

    # Ensure output is valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms

    return best_pts
