import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _negative_score(points_flat):
    points = points_flat.reshape(N, 2)
    score = _calculate_score(points)
    return -score

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _generate_k_ring(radii, counts, offsets):
    """
    Generate points in multiple concentric rings.
    radii: list of radii
    counts: list of number of points per ring
    offsets: list of angular offsets per ring
    """
    all_points = []
    for r, n, theta0 in zip(radii, counts, offsets):
        if n == 0:
            continue
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta0
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        all_points.append(np.column_stack((x, y)))
    if not all_points:
        return _generate_regular_polygon(1.0)
    return np.vstack(all_points)

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _polar_to_cartesian(polar):
    """polar: (N, 2) array of [radius, angle] -> (N, 2) cartesian"""
    r = polar[:, 0]
    theta = polar[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _cartesian_to_polar(points):
    """points: (N, 2) cartesian -> (N, 2) [radius, angle]"""
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack((r, theta))

def _polar_score(polar_flat):
    """
    Objective in polar coordinates.
    polar_flat: (2N,) array of [r0, t0, r1, t1, ...]
    Returns negative score (for minimization).
    """
    polar = polar_flat.reshape(N, 2)
    # Clamp radius to avoid negative/zero
    r = np.maximum(polar[:, 0], 0.01)
    theta = polar[:, 1]
    points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    score = _calculate_score(points)
    return -score

def _polish_cartesian(points, time_limit=5.0):
    """
    Polish points using L-BFGS-B on raw 2D coordinates.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    x0 = points.flatten()

    try:
        res = minimize(
            _negative_score,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if time.time() - start_time < time_limit:
            new_points = res.x.reshape(N, 2)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception:
        pass

    # Also try Nelder-Mead as a second polish
    if time.time() - start_time < time_limit - 1.0:
        try:
            res2 = minimize(
                _negative_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300, 'disp': False}
            )
            if time.time() - start_time < time_limit:
                new_points = res2.x.reshape(N, 2)
                new_score = _calculate_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    return best_score, best_points

def _polish_polar(polar_points, time_limit=5.0):
    """
    Polish in polar coordinates using L-BFGS-B.
    """
    start_time = time.time()
    best_score = _calculate_score(_polar_to_cartesian(polar_points))
    best_polar = polar_points.copy()
    x0 = polar_points.flatten()

    try:
        res = minimize(
            _polar_score,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        if time.time() - start_time < time_limit:
            new_polar = res.x.reshape(N, 2)
            new_points = _polar_to_cartesian(new_polar)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_polar = new_polar
    except Exception:
        pass

    return best_score, best_polar

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Phase 1: Regular polygon + polish
    if time.time() < deadline - 30:
        reg = _generate_regular_polygon(1.0)
        score, pts = _polish_cartesian(reg, time_limit=8.0)
        try_update(score, pts)

    # Phase 2: Multi-ring configurations
    ring_configs = [
        # (radii, counts, offsets)
        ([1.0], [14], [0.0]),
        ([1.0, 0.6], [8, 6], [0.0, np.pi/8]),
        ([1.0, 0.6], [7, 7], [0.0, np.pi/7]),
        ([1.0, 0.5], [9, 5], [0.0, np.pi/9]),
        ([1.0, 0.7], [10, 4], [0.0, np.pi/10]),
        ([1.0, 0.8, 0.4], [6, 4, 4], [0.0, np.pi/6, np.pi/4]),
        ([1.0, 0.5, 0.2], [7, 4, 3], [0.0, np.pi/7, np.pi/3]),
        ([1.0, 0.65], [12, 2], [0.0, np.pi/6]),
    ]

    for radii, counts, offsets in ring_configs:
        if time.time() > deadline - 25:
            break
        start_pts = _generate_k_ring(radii, counts, offsets)
        score, pts = _polish_cartesian(start_pts, time_limit=5.0)
        try_update(score, pts)

    # Phase 3: Differential evolution in polar coordinates (global search)
    if time.time() < deadline - 40:
        try:
            # Bounds for polar: radius in [0.1, 1.5], angle in [-pi, pi]
            bounds = [(0.1, 1.5), (-np.pi, np.pi)] * N

            def de_objective(x):
                if time.time() > deadline - 5:
                    return 1e6
                return _polar_score(x)

            # Use a moderate population size
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol',
                updating='immediate'
            )

            if time.time() < deadline - 20:
                de_polar = de_result.x.reshape(N, 2)
                de_points = _polar_to_cartesian(de_polar)
                # Polish the DE result
                score, pts = _polish_cartesian(de_points, time_limit=10.0)
                try_update(score, pts)

                # Also try polar polish on DE result
                score_p, polar_p = _polish_polar(de_polar, time_limit=5.0)
                pts_p = _polar_to_cartesian(polar_p)
                try_update(score_p, pts_p)

        except Exception:
            pass

    # Phase 4: Random starts with polish
    for seed in range(1, 15):
        if time.time() > deadline - 15:
            break
        start_pts = _generate_random_points(seed)
        score, pts = _polish_cartesian(start_pts, time_limit=3.0)
        try_update(score, pts)

    # Phase 5: Perturbed regular polygon with polish
    for noise_level in [0.05, 0.1, 0.2, 0.4]:
        if time.time() > deadline - 10:
            break
        rng = np.random.default_rng(100 + int(noise_level * 100))
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, noise_level, N)
        r = 1.0 + rng.normal(0, noise_level/5, N)
        r = np.maximum(r, 0.1)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        start_pts = np.column_stack((x, y))
        score, pts = _polish_cartesian(start_pts, time_limit=4.0)
        try_update(score, pts)

    # Final: ensure best_points is not None
    if best_points is None:
        best_points = _generate_regular_polygon(1.0)

    return best_points
