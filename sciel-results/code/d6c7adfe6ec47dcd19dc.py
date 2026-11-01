"""Covering the sphere with 50 equal caps – optimized via convex-hull facet targeting."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before hard limit


def _covering_radius(points):
    """
    Compute the covering radius (in degrees) of a set of points on the unit sphere.
    Uses convex hull: the covering radius equals the maximum circumradius of the
    facets of the convex hull (each facet's circumcenter on the sphere side).
    """
    points = np.asarray(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-12):
        return 180.0  # invalid
    points = points / norms

    # Check for coincident points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-8:
                return 180.0  # coincident points

    # Check not all in a closed hemisphere: origin should be inside convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0

    # Check if origin is inside the convex hull (i.e., points span all hemispheres)
    # A simple check: for each facet, the origin should be on the interior side
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Normal of the facet pointing outward
        if len(simplex) == 3:
            v1 = facet_points[1] - facet_points[0]
            v2 = facet_points[2] - facet_points[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            # Outward normal should point away from origin
            dot = np.dot(normal, facet_points[0])
            if dot < 0:
                normal = -normal
            # Check origin is on the interior side
            dist_origin = np.dot(normal, -facet_points[0])
            if dist_origin > 1e-6:
                # Origin is outside this facet's half-space -> all points in a hemisphere
                return 180.0
        elif len(simplex) == 2:
            # Edge case – shouldn't happen for 3D convex hull with 50 points
            pass

    # Compute circumradius for each facet
    max_r = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        if len(facet_pts) == 3:
            # Circumcenter of triangle on the sphere
            a, b, c = facet_pts
            # Solve for circumcenter: point equidistant from a, b, c on the sphere
            # Use the formula: circumcenter = (a + b + c) / |a + b + c| is not generally correct
            # Instead, solve the system:
            # |x|^2 = 1
            # x·a = x·b = x·c = cos(theta)
            # This gives: x·(a-b) = 0, x·(a-c) = 0, |x|^2 = 1
            # So x is perpendicular to (a-b) and (a-c), i.e., parallel to cross(a-b, a-c)
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                # Degenerate facet
                continue
            n = n / n_norm
            # The circumcenter on the sphere is in the direction of n (or -n)
            # We need the one that is on the "empty cap" side (away from the bulk)
            # The circumcenter should be on the side opposite to the centroid of the hull
            centroid = np.mean(facet_pts, axis=0)
            # Choose the sign so that the circumcenter is away from the origin-side
            # Actually, for a convex hull of points on the sphere, the circumcenter
            # of a facet lies on the sphere in the direction of the outward normal
            # if the facet is "acute" in some sense, otherwise on the opposite side.
            # Let's compute both and pick the one with larger angular distance to all points
            x1 = n
            x2 = -n
            # Angular distance from x to a, b, c
            d1 = np.arccos(np.clip(np.abs(np.dot(x1, a)), -1, 1))
            d2 = np.arccos(np.clip(np.abs(np.dot(x2, a)), -1, 1))
            # The circumradius is the angular distance from the circumcenter to any vertex
            # We want the circumcenter that is on the "empty" side, i.e., the side
            # where there are no other points. For a convex hull, the outward normal
            # points away from the hull interior.
            # The circumcenter on the sphere for the "empty cap" is the one in the
            # direction of the outward normal if the triangle is small enough.
            # Let's just use the maximum of the two possible circumradii? No.
            # Actually, the covering radius contribution from this facet is the
            # angular distance from the circumcenter (on the empty side) to the vertices.
            # The circumcenter on the empty side is the one farther from the other points.
            # For simplicity, compute the circumradius as the angular distance from
            # the circumcenter to vertex a, where the circumcenter is chosen to be
            # on the side of the outward normal.
            
            # Determine outward normal direction
            # The outward normal points away from the hull. Since all points are on
            # the unit sphere and the hull contains the origin, the outward normal
            # at a facet points away from the origin in the tangent plane sense.
            # Actually, for a point on the sphere, the outward normal of the hull
            # at that facet is in the direction that has positive dot product with
            # the facet centroid (since the centroid is in the direction of the facet).
            # Let's use: outward normal = n if n·centroid > 0 else -n
            if np.dot(n, centroid) > 0:
                outward = n
            else:
                outward = -n
            
            circumcenter = outward
            # Circumradius is the angular distance from circumcenter to any vertex
            r = np.arccos(np.clip(np.dot(circumcenter, a), -1, 1))
            max_r = max(max_r, r)
        elif len(facet_pts) == 2:
            # This shouldn't happen for a 3D convex hull with enough points
            continue
        else:
            # Higher-dimensional simplex (shouldn't happen in 3D)
            continue

    return np.degrees(max_r)


def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _objective(params):
    """Objective function: flatten params to points and compute covering radius."""
    points = params.reshape(N, 3)
    # Normalize each point
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-12):
        return 1e6
    points = points / norms
    return _covering_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    best_points = _initial_points()
    best_score = _covering_radius(best_points)

    # Local optimization using Nelder-Mead on the flattened parameter vector
    x0 = best_points.flatten()
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                new_points = result.x.reshape(N, 3)
                norms = np.linalg.norm(new_points, axis=1)
                if np.all(norms > 1e-12):
                    new_points = new_points / norms
                    best_points = new_points
                    best_score = result.fun
                x0 = new_points.flatten()
            else:
                # Perturb and try again
                x0 = best_points.flatten() + np.random.randn(N * 3) * 0.05
        except Exception:
            pass

        # Check time
        if time.time() > deadline:
            break

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms

    return best_points
