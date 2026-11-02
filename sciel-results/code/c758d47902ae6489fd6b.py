"""Heilbronn triangles: maximize the minimum triangle area for 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
# Vertices of the equilateral triangle with area 1
# Area of triangle with side s is (sqrt(3)/4) * s^2 = 1 => s = 4/sqrt(3)
SIDE = 4.0 / np.sqrt(3.0)
H = np.sqrt(3.0) / 2.0 * SIDE
V = np.array([[0.0, 0.0], [SIDE, 0.0], [SIDE / 2.0, H]])


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V[0], V[1])
    d2 = sign(p, V[1], V[2])
    d3 = sign(p, V[2], V[0])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _project_to_triangle(p):
    """Project point p to the nearest point inside the triangle."""
    if _inside_triangle(p):
        return p
    # Find closest point on triangle boundary
    best = None
    best_dist = np.inf
    for i in range(3):
        a = V[i]
        b = V[(i + 1) % 3]
        # Project p onto segment ab
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    return best


def _objective(x):
    """Objective to minimize: negative of min triangle area. x is 2N coordinates."""
    points = x.reshape(N, 2)
    # Penalty if any point is outside triangle
    for p in points:
        if not _inside_triangle(p):
            # Large penalty
            return 1e6
    return -_min_triangle_area(points)


def _soft_min_triangle_area(points, beta=50.0):
    """Soft approximation of min triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    return -(max_a + log_sum / beta)


def _objective_soft(x, beta=50.0):
    """Smooth objective for gradient-based optimization."""
    points = x.reshape(N, 2)
    return -_soft_min_triangle_area(points, beta)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # Initial guess: use a symmetric configuration
    # Place points at vertices, edge midpoints, and center
    # For 11 points in equilateral triangle:
    # 3 vertices + 3 edge midpoints + 1 center + 4 more points
    # Let's use a triangular lattice subset

    # Generate initial points using farthest point sampling
    rng = np.random.RandomState(42)

    # Start with vertices
    points = [V[0].copy(), V[1].copy(), V[2].copy()]

    # Add edge midpoints
    points.append((V[0] + V[1]) / 2.0)
    points.append((V[1] + V[2]) / 2.0)
    points.append((V[2] + V[0]) / 2.0)

    # Add centroid
    points.append(np.mean(V, axis=0))

    # Add 4 more points via farthest point sampling
    for _ in range(4):
        best_point = None
        best_min_dist = -1
        for _ in range(1000):
            # Random point in triangle
            u, v = rng.random(), rng.random()
            if u + v <= 1:
                p = V[0] + u * (V[1] - V[0]) + v * (V[2] - V[0])
            else:
                continue
            min_dist = min(np.linalg.norm(p - q) for q in points)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_point = p
        if best_point is not None:
            points.append(best_point)

    points = np.array(points[:N])

    # Ensure all points are inside triangle
    points = np.array([_project_to_triangle(p) for p in points])

    # Initial evaluation
    best_points = points.copy()
    best_score = _min_triangle_area(points)

    # Stage 1: Differential Evolution for global optimization
    if time.time() < deadline - 30:
        try:
            def de_obj(x):
                pts = x.reshape(N, 2)
                # Check if all points are inside
                for p in pts:
                    if not _inside_triangle(p):
                        return 1e6
                return -_min_triangle_area(pts)

            # Bounds for coordinates
            bounds = [(min(V[:, 0]) - 0.1, max(V[:, 0]) + 0.1),
                      (min(V[:, 1]) - 0.1, max(V[:, 1]) + 0.1)] * N

            # Run DE with limited iterations
            result = differential_evolution(
                de_obj,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False,
                init='sobol'
            )

            if result.success or result.fun < -best_score:
                cand_points = result.x.reshape(N, 2)
                # Project to triangle
                cand_points = np.array([_project_to_triangle(p) for p in cand_points])
                cand_score = _min_triangle_area(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass

    # Stage 2: Powell refinement
    if time.time() < deadline - 15:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            cand_points = result.x.reshape(N, 2)
            cand_points = np.array([_project_to_triangle(p) for p in cand_points])
            # Verify all inside
            valid = all(_inside_triangle(p) for p in cand_points)
            if valid:
                cand_score = _min_triangle_area(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass

    # Stage 3: Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            cand_points = result.x.reshape(N, 2)
            cand_points = np.array([_project_to_triangle(p) for p in cand_points])
            valid = all(_inside_triangle(p) for p in cand_points)
            if valid:
                cand_score = _min_triangle_area(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass

    # Final: ensure all points are strictly inside (with small margin)
    final_points = []
    for p in best_points:
        # Slightly shrink towards centroid if on boundary
        centroid = np.mean(V, axis=0)
        if not _inside_triangle(p):
            p = _project_to_triangle(p)
        # Add small margin
        p = p * 0.999 + centroid * 0.001
        if not _inside_triangle(p):
            p = centroid
        final_points.append(p)

    final_points = np.array(final_points)

    return final_points
