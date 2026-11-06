"""Pack 50 equal circles in the unit disk: soft-min L-BFGS-B + Newton polish + SA."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _raw_score(pts):
    """Compute the packing radius from centre positions."""
    if len(pts) != N or not np.all(np.isfinite(pts)):
        return -1.0
    # boundary
    dist_center = np.linalg.norm(pts, axis=1)
    r_boundary = 1.0 - dist_center
    if np.any(r_boundary < -1e-12):
        return -1.0
    # pairwise
    # Use a loop to avoid memory issues for N=50 (fine, but vectorized is faster)
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    r = min(min_pair / 2.0, np.min(r_boundary))
    return r


def _soft_min_objective(pts, beta):
    """
    Objective to minimize: -soft_min of all constraints.
    Constraints: for each pair (i,j), dist(i,j) >= 2r; for each i, 1-|c_i| >= r.
    We want to maximize r = min(all constraints).
    Use log-sum-exp: soft_min = -log(sum(exp(-x_i/beta))) * beta
    Minimize -soft_min = log(sum(exp(-x_i/beta))) * beta
    """
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    # Upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)

    # All constraint values (these are the "gaps")
    all_vals = np.concatenate([pair_dists / 2.0, boundary_dists])

    # soft_min via log-sum-exp
    # soft_min = -beta * log(sum(exp(-x / beta)))
    # We minimize -soft_min = beta * log(sum(exp(-x / beta)))
    # For numerical stability:
    x = -all_vals / beta
    x_max = np.max(x)
    log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    obj = beta * log_sum
    return obj


def _soft_min_gradient(pts, beta):
    """Gradient of the soft-min objective."""
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]

    boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
    all_vals = np.concatenate([pair_dists / 2.0, boundary_dists])

    x = -all_vals / beta
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    weights = exp_x / sum_exp  # softmax weights

    grad = np.zeros_like(pts)

    # Pairwise contribution: d/dx_i of (beta * log(sum(exp(-x_j/beta))))
    # x_j = dist(i,j)/2 for pair terms
    # d(dist/2)/d(pts_i) = (pts_i - pts_j) / (2 * dist)
    # The gradient of the objective w.r.t. pts_i:
    # sum over pairs j: weights_j * (-1/beta) * d(x_j)/d(pts_i) * beta
    # = -sum_j weights_j * d(x_j)/d(pts_i)

    n_pairs = N * (N - 1) // 2
    pair_weights = weights[:n_pairs]

    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        d = pair_dists[k]
        if d > 1e-12:
            direction = (pts[i] - pts[j]) / (2.0 * d)
            w = pair_weights[k]
            grad[i] -= w * direction
            grad[j] += w * direction

    # Boundary contribution
    boundary_weights = weights[n_pairs:]
    norms = np.linalg.norm(pts, axis=1)
    for i in range(N):
        if norms[i] > 1e-12:
            direction = pts[i] / norms[i]
            # x_i = 1 - |c_i|, d(x_i)/d(pts_i) = -pts_i/|c_i|
            # gradient contribution: -w_i * d(x_i)/d(pts_i) = w_i * pts_i/|c_i|
            grad[i] += boundary_weights[i] * direction

    return grad


def _newton_polish(pts, max_iter=200, tol=1e-12):
    """
    Newton-Raphson style polish: try to make the minimum constraint as large as possible
    by adjusting positions to satisfy contact constraints exactly.
    """
    pts = pts.copy()
    for it in range(max_iter):
        # Find the current minimum constraint
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        all_vals = np.concatenate([pair_dists / 2.0, boundary_dists])
        min_idx = np.argmin(all_vals)
        min_val = all_vals[min_idx]

        if min_val > 0.12:  # Already good enough
            break

        # Identify which constraint is binding
        n_pairs = N * (N - 1) // 2
        if min_idx < n_pairs:
            i, j = iu[0][min_idx], iu[1][min_idx]
            d = pair_dists[min_idx]
            if d < 1e-12:
                continue
            direction = (pts[i] - pts[j]) / d
            # Move i and j apart by half the deficit
            deficit = 2.0 * (0.125 - min_val)  # target radius approx
            step = deficit / 2.0
            pts[i] += direction * step
            pts[j] -= direction * step
        else:
            i = min_idx - n_pairs
            norm = np.linalg.norm(pts[i])
            if norm < 1e-12:
                continue
            direction = pts[i] / norm
            deficit = 0.125 - min_val
            pts[i] -= direction * deficit

        # Project back: ensure inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i] * 0.999

        if it % 10 == 0 and time.time() > TIME_LIMIT - 10:
            break

    return pts


def _make_rings():
    """Create a ring-based initial configuration."""
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    # Ring 1: 6 circles
    r1 = 0.2
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # Ring 2: 12 circles
    r2 = 0.4
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        pts[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    # Ring 3: 18 circles
    r3 = 0.6
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    # Ring 4: 14 circles
    r4 = 0.8
    for i in range(14):
        angle = 2 * np.pi * i / 14 + np.pi / 14
        pts[37 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return pts


def _hexagonal_seed():
    """Create a hexagonal lattice seed."""
    spacing = 0.25
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                points.append([x, y])
    points = np.array(points)
    # Sort by distance from center, take N
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    return points[idx]


def _random_seed():
    """Random initial configuration."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.9, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _refine(pts, start_time):
    """Refine a configuration using L-BFGS-B with decreasing beta."""
    best_pts = pts.copy()
    best_score = _raw_score(pts)

    betas = [0.05, 0.02, 0.01, 0.005, 0.002]
    for beta in betas:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        try:
            res = minimize(
                _soft_min_objective,
                pts,
                jac=_soft_min_gradient,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                new_pts = res.x
                new_score = _raw_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts.copy()
                    best_score = new_score
                pts = new_pts
        except Exception:
            pass

    # Newton polish
    if time.time() - start_time < TIME_LIMIT - 10:
        polished = _newton_polish(pts)
        polished_score = _raw_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score

    return best_pts, best_score


def _simulated_annealing(pts, start_time, max_time=60):
    """Simple simulated annealing to escape local minima."""
    best_pts = pts.copy()
    best_score = _raw_score(pts)
    current_score = best_score
    current_pts = pts.copy()

    T = 0.05
    T_min = 0.001
    cooling = 0.995
    step_size = 0.05

    while time.time() - start_time < start_time + max_time and T > T_min:
        # Perturb
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_pts = current_pts.copy()
        new_pts[idx] += delta

        # Keep inside boundary
        norm = np.linalg.norm(new_pts[idx])
        if norm > 1.0:
            new_pts[idx] = new_pts[idx] / norm * 0.999

        new_score = _raw_score(new_pts)
        delta_score = new_score - current_score

        if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_pts = current_pts.copy()
                best_score = current_score

        T *= cooling
        step_size *= 0.999

    return best_pts, best_score


def construct():
    start_time = time.time()

    # Try multiple initializations
    seeds = [
        _hexagonal_seed(),
        _make_rings(),
        _random_seed(),
        _random_seed(),
    ]

    best_pts = None
    best_score = -1.0

    for seed in seeds:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        refined, score = _refine(seed, start_time)
        if score > best_score:
            best_score = score
            best_pts = refined.copy()

    # If we have time, try SA on the best result
    if best_pts is not None and time.time() - start_time < TIME_LIMIT - 30:
        sa_pts, sa_score = _simulated_annealing(best_pts, start_time, max_time=TIME_LIMIT - 40)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()

    # Final polish
    if best_pts is not None and time.time() - start_time < TIME_LIMIT - 5:
        polished = _newton_polish(best_pts)
        polished_score = _raw_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score

    if best_pts is None:
        # Fallback
        best_pts = _hexagonal_seed()

    return best_pts
