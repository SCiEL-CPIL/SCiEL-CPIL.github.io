"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute convex hull area using monotone chain (Andrew's algorithm)."""
    n = len(points)
    if n < 3:
        return 0.0
    points = np.array(points, dtype=float)
    points = points[np.lexsort((points[:, 0], points[:, 1]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0

    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_area


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _regular_polygon(n, r=1.0):
    """Regular n-gon centered at origin with circumradius r."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _concentric_rings(n, n_rings=3):
    """Points distributed on concentric rings."""
    pts = []
    # Center
    pts.append(np.array([0.0, 0.0]))
    remaining = n - 1
    rings = min(n_rings, remaining)
    for ring in range(1, rings + 1):
        r = 0.3 * ring
        count = max(3, remaining // (rings - ring + 1)) if ring < rings else remaining
        angles = 2 * np.pi * np.arange(count) / count + ring * 0.3
        pts.extend([r * np.cos(a), r * np.sin(a)] for a in angles)
        remaining -= count
    # Fill remaining with random
    rng = np.random.RandomState(42)
    while len(pts) < n:
        pts.append(rng.randn(2) * 0.5)
    return np.array(pts[:n])


def _grid_init(n):
    """Approximate grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < n:
                pts.append([(i - side // 2) * 0.5, (j - side // 2) * 0.5])
    return np.array(pts)


def _random_init(n, seed):
    """Random initialization in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2)


def _optimize_from_start(start_pts, deadline, max_iter=5000):
    """Run Nelder-Mead optimization from a starting configuration."""
    x0 = start_pts.flatten()

    # Compute initial score
    initial_score = _score(start_pts)

    best_x = x0.copy()
    best_score = initial_score

    def objective(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)

    options = {
        'maxiter': max_iter,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'adaptive': True,
    }

    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        if result.fun < -best_score:
            best_score = -result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    best_pts = None
    best_score = 0.0

    # Generate diverse initial configurations
    initializations = []

    # Regular polygon
    initializations.append(_regular_polygon(N))

    # Concentric rings
    initializations.append(_concentric_rings(N))

    # Grid
    initializations.append(_grid_init(N))

    # Multiple random starts
    for seed in range(20):
        initializations.append(_random_init(N, seed))

    # Also try some structured variations
    # Perturbed regular polygon
    rng = np.random.RandomState(0)
    for _ in range(5):
        base = _regular_polygon(N)
        perturbed = base + rng.randn(N, 2) * 0.1
        initializations.append(perturbed)

    # Scale variations
    for scale in [0.5, 0.8, 1.2]:
        initializations.append(_regular_polygon(N, r=scale))

    # Optimize from each start
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break

        # Check if this init is already better
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts.copy()

        # Run optimization
        opt_pts, opt_score = _optimize_from_start(init_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

        # If we have good time, do a second polish pass
        if time.time() < deadline - 30 and opt_score > best_score:
            opt_pts2, opt_score2 = _optimize_from_start(opt_pts, deadline, max_iter=10000)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_pts = opt_pts2.copy()

    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N)

    return best_pts
