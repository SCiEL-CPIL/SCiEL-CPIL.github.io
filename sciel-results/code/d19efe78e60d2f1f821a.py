"""Pack 200 equal circles in a unit square using hex-grid initialization + L-BFGS-B soft-min optimization."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            D = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2))
            np.fill_diagonal(D, np.inf)
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_lattice(spacing, margin=0.02, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal lattice of candidate points inside [-0.5-margin, 0.5+margin]^2."""
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing

    pts = []
    y = -0.5 - margin + offset_y
    row_idx = 0
    while y <= 0.5 + margin:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 - margin + shift + offset_x
        while x <= 0.5 + margin:
            pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1

    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def _greedy_select(lattice_pts, N, target_r=None):
    """Greedy selection of N points from lattice_pts to maximize minimum distance."""
    M = len(lattice_pts)
    if M <= N:
        return lattice_pts.copy()

    selected = []
    selected_pts = np.empty((0, 2))
    available = list(range(M))
    available_set = set(range(M))

    # Precompute wall distances for each lattice point
    wall_dists = np.minimum(
        np.minimum(lattice_pts[:, 0] + 0.5, 0.5 - lattice_pts[:, 0]),
        np.minimum(lattice_pts[:, 1] + 0.5, 0.5 - lattice_pts[:, 1])
    )

    # Start with the point that has the best wall distance (most central)
    best_idx = int(np.argmax(wall_dists))
    selected.append(best_idx)
    selected_pts = lattice_pts[best_idx:best_idx+1].copy()
    available_set.discard(best_idx)
    available.remove(best_idx)

    for count in range(1, N):
        if time.time() > _TIME_LIMIT - 15:
            break

        # Build KDTree from selected points for efficient distance queries
        if len(selected_pts) > 0:
            tree = KDTree(selected_pts)
        else:
            tree = None

        best_idx = None
        best_score = -1

        for idx in available:
            # Check wall constraint
            wd = wall_dists[idx]
            if target_r is not None and wd < target_r:
                continue

            # Check distance to already selected points
            if tree is not None and len(selected_pts) > 0:
                dist, _ = tree.query(lattice_pts[idx])
                if target_r is not None and dist < 2 * target_r:
                    continue
                score = min(wd, dist / 2.0)
            else:
                score = wd

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # Fallback: pick the point with best wall distance
            remaining = list(available_set)
            if remaining:
                best_idx = max(remaining, key=lambda i: wall_dists[i])
            else:
                break

        selected.append(best_idx)
        selected_pts = np.vstack([selected_pts, lattice_pts[best_idx:best_idx+1]])
        available_set.discard(best_idx)
        available.remove(best_idx)

    return lattice_pts[selected]


def _soft_min_value_and_grad(pts, beta):
    """
    Compute soft-min of (pairwise distances/2, wall distances) and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros_like(pts)

    # Wall distances: 4 per point (left, right, bottom, top)
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair distances: use KDTree to find nearby pairs
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)

    if len(pairs) == 0:
        pair_vals = np.array([10.0])
    else:
        pair_vals = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])

    all_vals = np.concatenate([wall_vals, pair_vals])

    # Soft min computation: soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    soft_min_val = -(max_neg + np.log(sum_exp)) / beta

    # Gradient
    grad = np.zeros_like(pts)
    weights = exp_shifted / sum_exp

    # Wall gradients
    for i in range(n):
        grad[i, 0] += weights[4*i] * 1.0
        grad[i, 0] += weights[4*i+1] * (-1.0)
        grad[i, 1] += weights[4*i+2] * 1.0
        grad[i, 1] += weights[4*i+3] * (-1.0)

    # Pair gradients
    if len(pairs) > 0:
        for k, (i, j) in enumerate(pairs):
            w = weights[4*n + k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[i] += w * diff / (2.0 * dist)
            grad[j] -= w * diff / (2.0 * dist)

    return soft_min_val, grad


def _objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B: minimize -soft_min to maximize soft_min."""
    pts = x.reshape(-1, 2)
    val, grad = _soft_min_value_and_grad(pts, beta)
    # We want to maximize soft_min, so minimize -soft_min
    return -val, -grad.reshape(-1)


def _smooth_polish(pts, max_iters=5000, lr=0.0003, cooling=0.9998, beta_start=10.0, beta_end=300.0):
    """
    Gradient-based smooth optimization to polish the packing using soft-min objective.
    """
    pts = pts.copy()

    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
            break

        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress

        obj, grad = _soft_min_value_and_grad(pts, beta)

        pts += lr * grad
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

        lr *= cooling

        if np.max(np.abs(grad)) < 1e-9:
            break

    return pts


def _lbfgs_polish(pts, beta_start=10.0, beta_end=200.0, max_time=60.0):
    """
    Use L-BFGS-B to optimize the soft-min objective with increasing beta.
    """
    pts = pts.copy()
    t0 = time.time()
    n = len(pts)
    x0 = pts.flatten()

    # Bounds: keep points inside the square
    bounds = [(-0.5, 0.5)] * (2 * n)

    # Phase 1: Lower beta for broad optimization
    beta1 = beta_start
    if time.time() - t0 < max_time - 10:
        res1 = minimize(
            _objective_and_grad,
            x0,
            args=(beta1,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res1.success:
            pts = res1.x.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    # Phase 2: Higher beta for sharper min
    beta2 = beta_end
    if time.time() - t0 < max_time - 5:
        x1 = pts.flatten()
        res2 = minimize(
            _objective_and_grad,
            x1,
            args=(beta2,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res2.success:
            pts = res2.x.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    return pts


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Strategy:
    # 1. Try multiple hex lattice configurations with different spacings and offsets
    # 2. Greedy select N points from each lattice
    # 3. Polish with L-BFGS-B on soft-min objective
    # 4. Keep the best result

    # Different lattice spacings and offsets to try
    # For 200 circles in unit square, typical spacing ~ 1/sqrt(200) ≈ 0.07
    configs = [
        (0.065, 0.0, 0.0),
        (0.070, 0.0, 0.0),
        (0.075, 0.0, 0.0),
        (0.065, 0.02, 0.01),
        (0.070, 0.02, 0.01),
        (0.075, 0.02, 0.01),
        (0.068, 0.03, 0.02),
        (0.072, 0.03, 0.02),
    ]

    for spacing, ox, oy in configs:
        if time.time() - t0 > _TIME_LIMIT - 40:
            break

        # Generate hex lattice
        lattice = _generate_hex_lattice(spacing, margin=0.02, offset_x=ox, offset_y=oy)

        if len(lattice) < N:
            continue

        # Greedy selection
        selected = _greedy_select(lattice, N)

        if len(selected) < N:
            continue

        r_init = _compute_radius(selected)

        # Polish with L-BFGS-B
        if time.time() - t0 < _TIME_LIMIT - 25:
            polished = _lbfgs_polish(selected, beta_start=10.0, beta_end=150.0,
                                     max_time=min(30.0, _TIME_LIMIT - 25 - (time.time() - t0)))
            r_polished = _compute_radius(polished)
        else:
            polished = selected
            r_polished = r_init

        # Keep the better of initial and polished
        if r_polished >= r_init:
            current_r, current_pts = r_polished, polished
        else:
            current_r, current_pts = r_init, selected

        if current_r > best_r:
            best_r = current_r
            best_pts = current_pts.copy()

    # Final high-precision L-BFGS-B polish on the best result
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 15:
        final_polish = _lbfgs_polish(best_pts, beta_start=30.0, beta_end=300.0,
                                     max_time=min(20.0, _TIME_LIMIT - 15 - (time.time() - t0)))
        r_final = _compute_radius(final_polish)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_polish.copy()

    # Fallback
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]

    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
