# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize


def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def objective(x):
    """
    Objective function for minimization.
    We want to maximize (d_min / d_max)^2, so we minimize its negative.
    x is a flat array of shape (32,).
    """
    points = x.reshape(-1, 2)
    score = calculate_score(points)
    return -score


def hexagonal_init(n, rotation=0.0, scale=1.0):
    """Generate a hexagonal grid initialization."""
    points = []
    # 4x4 grid is 16 points
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    # Center
    center = pts.mean(axis=0)
    pts = pts - center
    # Rotate
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    # Normalize to roughly fit in [-1, 1]
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * scale
    return pts


def random_init(n, seed=42, scale=1.0):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    # Random points in unit circle
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y)) * scale


def square_init(n, scale=1.0):
    """Generate a square grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j * scale, i * scale])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * scale
    return pts


def circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))


def concentric_circles_init(n, scale=1.0):
    """Generate points on two concentric circles (8 each)."""
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    c1 = np.column_stack([np.cos(angles1), np.sin(angles1)]) * scale
    c2 = np.column_stack([np.cos(angles2), np.sin(angles2)]) * (scale * 0.5)
    return np.vstack([c1, c2])


def repulsive_relaxation(points, start_time, max_time=580.0):
    """
    Relax points using repulsive forces with cooling schedule.
    Points repel each other with inverse-square force, encouraging uniform spacing.
    """
    points = points.copy().astype(float)
    n = len(points)

    # Normalize initial scale
    dists = spd.pdist(points)
    current_dmax = np.max(dists)
    if current_dmax > 0:
        points = points / current_dmax

    # Parameters
    temp = 1.0
    cooling = 0.995
    min_temp = 1e-8
    max_iter = 100000

    for iteration in range(max_iter):
        # Check time budget
        if time.time() - start_time > max_time:
            break

        # Compute pairwise forces
        forces = np.zeros_like(points)

        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                force_mag = 1.0 / (dist * dist + 0.01)
                force_vec = diff / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

        # Center of mass correction
        forces -= np.mean(forces, axis=0)

        # Update positions
        step_size = temp
        points += forces * step_size

        # Normalize to keep dmax ~ 1
        dists = spd.pdist(points)
        dmax = np.max(dists)
        if dmax > 0:
            points = points / dmax

        # Cool down
        temp *= cooling
        if temp < min_temp:
            break

    return points


def local_refinement(points, start_time, max_time=580.0):
    """
    Local refinement using small random perturbations accepted if they improve score.
    """
    points = points.copy().astype(float)
    n = len(points)
    current_score = calculate_score(points)

    # Normalize
    dists = spd.pdist(points)
    dmax = np.max(dists)
    if dmax > 0:
        points = points / dmax
        current_score = calculate_score(points)

    temp = 0.01
    cooling = 0.99
    min_temp = 1e-10

    while time.time() - start_time < max_time:
        if temp < min_temp:
            break

        idx = np.random.randint(n)
        delta = np.random.randn(2) * temp
        new_points = points.copy()
        new_points[idx] += delta

        dists = spd.pdist(new_points)
        dmax = np.max(dists)
        if dmax > 0:
            new_points = new_points / dmax

        new_score = calculate_score(new_points)

        if new_score >= current_score:
            points = new_points
            current_score = new_score
        else:
            prob = np.exp((new_score - current_score) / max(temp, 1e-10))
            if np.random.random() < prob:
                points = new_points
                current_score = new_score

        temp *= cooling

    return points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit
    n = 16

    # Generate diverse initializations
    initializations = []

    # Hexagonal grids with various rotations
    for rot in [0, np.pi / 6, np.pi / 3, np.pi / 2]:
        initializations.append(hexagonal_init(n, rotation=rot))

    # Square grid
    initializations.append(square_init(n))

    # Points on a circle
    initializations.append(circle_init(n))

    # Two concentric circles
    initializations.append(concentric_circles_init(n))

    # Random points
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(random_init(n, seed=seed))

    # Jittered hexagonal grids
    rng = np.random.RandomState(123)
    for _ in range(5):
        base = hexagonal_init(n, rotation=rng.rand() * np.pi)
        jitter = rng.randn(n, 2) * 0.1
        initializations.append(base + jitter)

    best_points = None
    best_score = -1.0

    # Phase 1: Powell optimization for each initialization
    # We allocate a fixed time budget per initialization
    num_starts = len(initializations)
    time_per_start = 15.0  # seconds

    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > max_time - 60:
            break

        x0 = init_pts.flatten()

        try:
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )

            if res.success or res.fun < 0:
                points_opt = res.x.reshape(-1, 2)
                score_opt = -res.fun

                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt

        except Exception:
            pass

    # Fallback if nothing found
    if best_points is None:
        best_points = hexagonal_init(n)
        best_score = calculate_score(best_points)

    # Phase 2: Refinement with repulsive relaxation and local search on the best point
    if time.time() - start_time < max_time - 30:
        # Try repulsive relaxation on the best point found
        relaxed = repulsive_relaxation(best_points, start_time, max_time - 30)
        score_relaxed = calculate_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_points = relaxed

    # Phase 3: Final local refinement (SA) if time permits
    if time.time() - start_time < max_time - 5:
        refined = local_refinement(best_points, start_time, max_time - 5)
        score_refined = calculate_score(refined)
        if score_refined > best_score:
            best_score = score_refined
            best_points = refined

    return best_points


# EVOLVE-BLOCK-END
