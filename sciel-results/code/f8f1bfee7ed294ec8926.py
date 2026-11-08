"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area if all points are on the hull
        from scipy.spatial import ConvexHull
        # If fewer than 3 points, area is 0
        if len(points) < 3:
            return 0.0
        # Compute area using the shoelace formula on the hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        n = len(hull_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull_points[i][0] * hull_points[j][1]
            area -= hull_points[j][0] * hull_points[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Start with regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    # Try multiple optimization runs with different initial conditions
    rng = np.random.default_rng(42)

    # Objective: minimize negative score (i.e., maximize score)
    # But we need to handle the convex hull normalization carefully.
    # Let's optimize the raw minimum triangle area with a fixed normalization,
    # or better, optimize the points directly to maximize min_area/hull_area.

    # Since the score is scale-invariant (both min_area and hull_area scale with size^2),
    # we can fix the convex hull area to 1 and maximize the minimum triangle area.
    # Alternatively, we can just optimize the points and compute the ratio.

    # Let's use a different approach: optimize points on a unit circle or in a unit square
    # and then normalize.

    # Actually, let's just do local optimization from multiple starting points.

    for restart in range(10):
        if time.time() > deadline:
            break

        if restart == 0:
            points0 = best_points.copy()
        else:
            # Random perturbation of the regular polygon
            points0 = _regular_polygon(N)
            points0 += rng.normal(0, 0.1, size=points0.shape)

        # Define objective: minimize negative score
        def objective(x):
            pts = x.reshape(-1, 2)
            # Ensure convex hull area is reasonable
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            min_area = _min_triangle_area(pts)
            return -min_area / hull_area

        # Use Nelder-Mead for robustness
        res = minimize(objective, points0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})

        if time.time() > deadline:
            break

        if res.success or res.fun < -best_score:
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

    return best_points
# EVOLVE-BLOCK-END
