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


def smooth_objective(x, beta=8.0):
    """
    Smooth objective using log-sum-exp for soft-min.
    We fix the scale by normalizing so d_max = 1.
    Then maximize d_min^2, which is equivalent to maximizing d_min.
    We minimize -d_min^2 via soft-min surrogate.
    """
    points = x.reshape(-1, 2)
    # Normalize: divide by d_max
    dists = spd.pdist(points)
    d_max = np.max(dists)
    if d_max < 1e-12:
        return 0.0
    points = points / d_max
    dists = spd.pdist(points)
    
    # Soft min via log-sum-exp
    # soft_min = -1/beta * log(sum(exp(-beta * d)))
    # Shift by min for numerical stability
    d_min_raw = np.min(dists)
    shifted = dists - d_min_raw
    log_sum = d_min_raw + np.log(np.sum(np.exp(-beta * shifted))) / beta
    # Actually: soft_min = d_min_raw - (1/beta)*log(sum(exp(-beta*(d - d_min_raw))))
    # Wait: -1/beta * log(sum(exp(-beta*d))) 
    # = -1/beta * log(exp(-beta*d_min) * sum(exp(-beta*(d-d_min))))
    # = d_min - 1/beta * log(sum(exp(-beta*(d-d_min))))
    soft_min = d_min_raw - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    # We want to maximize soft_min^2 (since d_max is fixed at 1)
    # So minimize -soft_min^2
    return -(soft_min * soft_min)


def smooth_gradient(x, beta=8.0):
    """
    Compute gradient of smooth_objective analytically.
    """
    points = x.reshape(-1, 2)
    dists = spd.pdist(points)
    d_max = np.max(dists)
    if d_max < 1e-12:
        return np.zeros_like(x)
    
    points = points / d_max
    dists = spd.pdist(points)
    
    # soft_min = d_min - (1/beta) * log(sum(exp(-beta * (d - d_min))))
    d_min_raw = np.min(dists)
    shifted = dists - d_min_raw
    exp_neg_beta_shifted = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta_shifted)
    
    # soft_min = d_min - (1/beta) * log(sum_exp)
    # d(soft_min)/d(d_k) = -1/beta * exp(-beta*(d_k - d_min)) / sum_exp  (for each pair k)
    # But d_k depends on points...
    
    # Actually, let's just use finite differences for the gradient since 
    # the analytic gradient is complex. But that defeats the purpose.
    
    # Alternative: use the fact that we're optimizing -soft_min^2
    # and use scipy's minimize with numerical gradient but better initialization.
    
    # For now, return zero and let minimize compute gradient numerically
    return np.zeros_like(x)


def hexagonal_init(n, rotation=0.0, scale=1.0):
    """Generate a hexagonal grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * scale
    return pts


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


def golden_ratio_init(n, scale=1.0):
    """Generate points using golden ratio spiral."""
    points = []
    phi = (1 + np.sqrt(5)) / 2
    for i in range(n):
        r = np.sqrt(i / n) * scale
        theta = 2 * np.pi * i / phi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def repulsive_relaxation(points, start_time, max_time=580.0):
    """
    Relax points using repulsive forces with cooling schedule.
    """
    points = points.copy().astype(float)
    n = len(points)

    dists = spd.pdist(points)
    current_dmax = np.max(dists)
    if current_dmax > 0:
        points = points / current_dmax

    temp = 1.0
    cooling = 0.995
    min_temp = 1e-8
    max_iter = 100000

    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break

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

        forces -= np.mean(forces, axis=0)

        step_size = temp
        points += forces * step_size

        dists = spd.pdist(points)
        dmax = np.max(dists)
        if dmax > 0:
            points = points / dmax

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


def nelder_mead_polish(points, start_time, max_time=580.0):
    """
    High-precision polish using Nelder-Mead with tight tolerances.
    """
    x0 = points.flatten()
    try:
        res = minimize(
            lambda x: -calculate_score(x.reshape(-1, 2)),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        points_opt = res.x.reshape(-1, 2)
        return points_opt
    except Exception:
        return points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    max_time = 590.0  # Safety margin
    n = 16

    # Generate diverse initializations
    initializations = []

    for rot in [0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi / 12]:
        initializations.append(hexagonal_init(n, rotation=rot))

    initializations.append(square_init(n))
    initializations.append(circle_init(n))
    initializations.append(concentric_circles_init(n))
    initializations.append(golden_ratio_init(n))

    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = np.random.RandomState(seed)
        r = rng.rand(n) ** 0.5
        theta = rng.rand(n) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        initializations.append(np.column_stack((x, y)))

    rng = np.random.RandomState(123)
    for _ in range(8):
        base = hexagonal_init(n, rotation=rng.rand() * np.pi)
        jitter = rng.randn(n, 2) * 0.05
        initializations.append(base + jitter)

    best_points = None
    best_score = -1.0

    # Phase 1: Powell optimization for each initialization
    time_per_start = 12.0

    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > max_time - 120:
            break

        x0 = init_pts.flatten()

        try:
            res = minimize(
                lambda x: -calculate_score(x.reshape(-1, 2)),
                x0,
                method='Powell',
                options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11}
            )

            points_opt = res.x.reshape(-1, 2)
            score_opt = calculate_score(points_opt)

            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt

        except Exception:
            pass

    if best_points is None:
        best_points = hexagonal_init(n)
        best_score = calculate_score(best_points)

    # Phase 2: Smooth objective optimization with L-BFGS-B
    if time.time() - start_time < max_time - 60:
        for beta_val in [5.0, 8.0, 12.0]:
            if time.time() - start_time > max_time - 60:
                break
            try:
                x0 = best_points.flatten()
                res = minimize(
                    lambda x: smooth_objective(x, beta=beta_val),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 100000}
                )
                points_opt = res.x.reshape(-1, 2)
                score_opt = calculate_score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt
            except Exception:
                pass

    # Phase 3: Nelder-Mead polish
    if time.time() - start_time < max_time - 30:
        polished = nelder_mead_polish(best_points, start_time, max_time - 30)
        score_polished = calculate_score(polished)
        if score_polished > best_score:
            best_score = score_polished
            best_points = polished

    # Phase 4: Repulsive relaxation
    if time.time() - start_time < max_time - 15:
        relaxed = repulsive_relaxation(best_points, start_time, max_time - 15)
        score_relaxed = calculate_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_points = relaxed

    # Phase 5: Final local refinement
    if time.time() - start_time < max_time - 5:
        refined = local_refinement(best_points, start_time, max_time - 5)
        score_refined = calculate_score(refined)
        if score_refined > best_score:
            best_score = score_refined
            best_points = refined

    return best_points


# EVOLVE-BLOCK-END
