"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill


def _validate(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _total_radius(circles):
    return float(np.sum(circles[:, 2]))


def _smooth_objective(params, W, H, beta=50.0):
    """
    Smooth objective: maximize sum(r) - soft penalties for overlaps and boundary.
    Uses log-sum-exp for smooth max(0, violation) terms.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Soft penalty for negative radii
    neg_r = np.maximum(0.0, -r)
    penalty_r = beta * np.logsumexp(neg_r / beta) if np.any(neg_r > 0) else 0.0

    # Boundary violations
    viol_left = np.maximum(0.0, r - x)
    viol_right = np.maximum(0.0, r + x - W)
    viol_bottom = np.maximum(0.0, r - y)
    viol_top = np.maximum(0.0, r + y - H)

    boundary_viol = np.concatenate([viol_left, viol_right, viol_bottom, viol_top])
    penalty_boundary = beta * np.logsumexp(boundary_viol / beta) if np.any(boundary_viol > 0) else 0.0

    # Overlap violations: for each pair (i,j), violation = r_i + r_j - dist_ij
    overlap_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            if viol > 0:
                overlap_violations.append(viol)

    if overlap_violations:
        ov = np.array(overlap_violations)
        penalty_overlap = beta * np.logsumexp(ov / beta)
    else:
        penalty_overlap = 0.0

    # Objective: minimize -sum(r) + weighted penalties
    obj = -np.sum(r) + 10.0 * penalty_overlap + 100.0 * penalty_boundary + 100.0 * penalty_r
    return obj


def _relax_circles(circles, W, H, max_iter=2000, beta=50.0):
    """
    Optimize circle positions and radii using SLSQP with smooth objective.
    """
    params = circles.flatten()
    n = N

    def obj(p):
        return _smooth_objective(p, W, H, beta)

    # Bounds: x in [0, W], y in [0, H], r in [0.001, min(W,H)/2]
    r_max = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))      # x
        bounds.append((0.0, H))      # y
        bounds.append((0.001, r_max))  # r

    result = minimize(
        obj, params,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
    )

    new_circles = result.x.reshape(n, 3)
    # Clip to ensure validity
    new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], W - new_circles[:, 2])
    new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], H - new_circles[:, 2])
    new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, r_max)

    return new_circles


def _nelder_mead_polish(circles, W, H, max_iter=3000):
    """Final polish with Nelder-Mead for high precision."""
    params = circles.flatten()

    def obj(p):
        return _smooth_objective(p, W, H, beta=200.0)

    result = minimize(
        obj, params,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )

    new_circles = result.x.reshape(N, 3)
    r_max = min(W, H) / 2.0
    new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], W - new_circles[:, 2])
    new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], H - new_circles[:, 2])
    new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, r_max)

    return new_circles


def _grid_init(W, H):
    """Grid-based initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2.0 * side) * 0.95
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_init(W, H, rng):
    """Random initialization with small circles."""
    circles = []
    r_max = min(W, H) / 5.0
    for _ in range(N):
        r = rng.uniform(0.005, r_max)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _hex_init(W, H):
    """Hexagonal lattice initialization."""
    # For N=21, use a hex arrangement: rows of 3,4,4,4,4,2 or similar
    # Let's try a more structured approach
    # 21 = 3+4+4+4+4+2 or 4+4+4+4+3+2
    row_counts = [3, 4, 4, 4, 4, 2]
    circles = []
    n_rows = len(row_counts)
    row_h = H / n_rows
    for row_idx, count in enumerate(row_counts):
        cy = (row_idx + 0.5) * row_h
        for col_idx in range(count):
            if len(circles) >= N:
                break
            # Stagger odd rows
            offset = 0.5 * (W / count) if (row_idx % 2 == 1) else 0.0
            cx = (col_idx + 0.5 + offset) * W / count
            # Limit cx to valid range
            cx = np.clip(cx, 0.1, W - 0.1)
            r = min(row_h / 2.0, W / (count * 1.2)) * 0.85
            circles.append([cx, cy, r])
    return np.array(circles[:N])


def _pack_for_aspect(W, H, rng, time_budget):
    """Try multiple initializations and optimize for a given aspect ratio."""
    best_circles = None
    best_score = -np.inf

    initializations = [
        _grid_init(W, H),
        _hex_init(W, H),
    ]

    # Add random initializations
    n_random = 3
    for _ in range(n_random):
        initializations.append(_random_init(W, H, rng))

    for init_circles in initializations:
        if time.time() > time_budget:
            break

        # SLSQP optimization
        opt_circles = _relax_circles(init_circles, W, H, max_iter=1000, beta=50.0)

        # Check validity and score
        if _validate(opt_circles, W, H):
            score = _total_radius(opt_circles)
            if score > best_score:
                best_score = score
                best_circles = opt_circles.copy()

        # Polish with Nelder-Mead if time permits
        if time.time() < time_budget - 5:
            polished = _nelder_mead_polish(opt_circles, W, H, max_iter=1500)
            if _validate(polished, W, H):
                score = _total_radius(polished)
                if score > best_score:
                    best_score = score
                    best_circles = polished.copy()

    return best_circles, best_score


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Perimeter = 2(W+H) = 4 => W + H = 2
    # Try a range of aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Time budget for this aspect ratio
        remaining = DEADLINE - elapsed - 5
        if remaining < 5:
            break

        # Allocate time: give more time to promising aspect ratios
        # Square (ar=1) and near-square are likely good
        if 0.7 <= ar <= 1.5:
            time_budget = time.time() + min(remaining * 0.3, 20.0)
        else:
            time_budget = time.time() + min(remaining * 0.15, 10.0)

        circles, score = _pack_for_aspect(W, H, rng, time_budget)

        if circles is not None and score > best_score:
            best_score = score
            best_result = circles.copy()

    # If we have time, do additional random restarts on the best aspect ratio
    if best_result is not None and time.time() < DEADLINE - 15:
        # Try a few more random restarts
        n_extra = 5
        for i in range(n_extra):
            if time.time() > DEADLINE - 5:
                break
            # Use the best aspect ratio (infer from best_result)
            # Actually, let's just try square and a couple others
            for ar in [0.8, 1.0, 1.2]:
                if time.time() > DEADLINE - 5:
                    break
                W = 2.0 * ar / (ar + 1.0)
                H = 2.0 / (ar + 1.0)
                circ_rand = _random_init(W, H, rng)
                opt_circ = _relax_circles(circ_rand, W, H, max_iter=800, beta=30.0)
                if _validate(opt_circ, W, H):
                    score = _total_radius(opt_circ)
                    if score > best_score:
                        best_score = score
                        best_result = opt_circ.copy()

    # Fallback: safe grid in unit square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    # Final validation
    if not _validate(best_result, 1.0, 1.0):
        # Need to check with the actual W, H used
        # For safety, use a very conservative grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.8)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
