"""Optimized program for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
DEADLINE = 150.0
START_TIME = time.time()


def _convex_hull_area(points):
    """Compute area of convex hull of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _soft_min_triangle_area(points, beta=50.0):
    """Soft-min surrogate for minimum triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min via log-sum-exp: -log(sum(exp(-beta * x))) / beta
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta


def _soft_score(points, beta=50.0):
    """Soft score surrogate."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    soft_min = _soft_min_triangle_area(points, beta)
    return soft_min / hull_area


def _negative_soft_score(params, beta=50.0):
    """Negative soft score for minimization."""
    n = len(params)
    points = params.reshape((-1, 2))
    return -_soft_score(points, beta)


def _make_two_ring_config(r_inner, r_outer, angle_offset, n_inner=7, n_outer=7):
    """Create a two-ring configuration."""
    points = []
    # Inner ring
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + angle_offset
        points.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    # Outer ring
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer + angle_offset
        points.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    return np.array(points)


def construct():
    """Construct 14 points to maximize min triangle area / convex hull area."""
    best_points = None
    best_score = -1.0
    deadline = START_TIME + DEADLINE - 5.0  # Safety margin

    # Try different two-ring configurations as starting points
    r_inner_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
    r_outer_candidates = [1.0, 1.1, 1.2]
    angle_offsets = [0.0, np.pi / 14, np.pi / 7, 2 * np.pi / 14]

    starting_configs = []
    for r_in in r_inner_candidates:
        for r_out in r_outer_candidates:
            for offset in angle_offsets:
                points = _make_two_ring_config(r_in, r_out, offset)
                starting_configs.append(points)

    # Also add some random configurations
    np.random.seed(42)
    for _ in range(10):
        points = np.random.randn(N, 2) * 0.5
        points += np.random.randn(2)  # Random center
        starting_configs.append(points)

    # Optimize each starting configuration
    for idx, init_points in enumerate(starting_configs):
        if time.time() > deadline:
            break

        # Normalize initial points to have reasonable scale
        center = np.mean(init_points, axis=0)
        init_points = init_points - center
        max_norm = np.max(np.linalg.norm(init_points, axis=1))
        if max_norm > 1e-10:
            init_points = init_points / max_norm * 1.0

        params = init_points.flatten()

        # Try different beta values for soft-min
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    _negative_soft_score,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )

                opt_points = result.x.reshape((-1, 2))
                # Center the points
                center = np.mean(opt_points, axis=0)
                opt_points = opt_points - center

                # Compute actual score
                actual_score = _score(opt_points)

                if actual_score > best_score:
                    best_score = actual_score
                    best_points = opt_points.copy()

            except Exception:
                continue

    # If no valid result, return a default configuration
    if best_points is None:
        best_points = _make_two_ring_config(0.5, 1.0, 0.0)
        center = np.mean(best_points, axis=0)
        best_points = best_points - center

    return best_points


# Test at module level
if __name__ == "__main__":
    pts = construct()
    print("Score:", _score(pts))
    print("Points shape:", pts.shape)
