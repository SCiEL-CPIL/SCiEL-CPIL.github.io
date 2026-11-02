"""Rewritten program: differentiable rigid-body assembly + LP radius sub-problem."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

DEADLINE = 115.0  # safety margin before 120s kill


def _solve_radii(positions, w, h):
    """
    Given fixed positions and rectangle (w, h), solve LP:
    max sum r_i
    s.t. r_i + r_j <= dist(i,j) - eps  for all i<j
         r_i <= min(x_i, h - y_i) - eps  for all i
         r_i >= 0
    """
    n = len(positions)
    eps = 1e-4
    m = n * (n - 1) // 2 + n
    # objective: maximize sum r_i -> minimize -sum r_i
    c = -np.ones(n)
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)

    row = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            A_ub[row, i] = 1.0
            A_ub[row, j] = 1.0
            b_ub[row] = d - eps
            row += 1

    for i in range(n):
        wall_dist = min(positions[i][0], h - positions[i][1], positions[i][1], w - positions[i][0])
        # Actually: x_i >= r_i, h - y_i >= r_i, y_i >= r_i, w - x_i >= r_i
        # so r_i <= min(x_i, h - y_i, y_i, w - x_i)
        wall_dist = min(positions[i][0], h - positions[i][1], positions[i][1], w - positions[i][0])
        A_ub[row, i] = 1.0
        b_ub[row] = wall_dist - eps
        row += 1

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
    if res.success:
        return -res.fun, res.x
    else:
        # Fallback: use min wall distance / 2 as conservative radius
        radii = np.array([min(pos[0], h - pos[1], pos[1], w - pos[0]) * 0.49 for pos in positions])
        return radii.sum(), radii


def _check_validity(positions, radii, w, h, tol=1e-3):
    """Check if circles are disjoint and inside rectangle."""
    n = len(positions)
    for i in range(n):
        if positions[i][0] - radii[i] < -tol or positions[i][0] + radii[i] > w + tol:
            return False
        if positions[i][1] - radii[i] < -tol or positions[i][1] + radii[i] > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _initial_positions(n, w, h, seed=0):
    """Grid-like initial positions with jitter."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else n
    rows = int(np.ceil(n / cols))
    positions = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = (c + 0.5) * w / cols
        y = (r + 0.5) * h / rows
        x += rng.uniform(-0.02 * w, 0.02 * w)
        y += rng.uniform(-0.02 * h, 0.02 * h)
        positions.append([x, y])
    return np.array(positions)


def _gradient_descent(positions, w, h, max_iters=5000, lr=0.001, seed=0):
    """
    Optimize positions to maximize sum of radii (approximated via LP at each step is too slow).
    Instead, use a smooth surrogate: maximize sum of min-pairwise-distances and wall-distances.
    We use a simple gradient ascent on a smooth objective.
    """
    rng = np.random.default_rng(seed)
    positions = positions.copy()
    n = len(positions)
    best_obj = -np.inf
    best_pos = positions.copy()

    for it in range(max_iters):
        # Compute objective: sum over i of min(wall_dist_i, min_j dist(i,j))
        obj = 0.0
        grad = np.zeros_like(positions)
        for i in range(n):
            wall_d = min(positions[i][0], h - positions[i][1], positions[i][1], w - positions[i][0])
            min_pair = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_pair:
                    min_pair = d
            val = min(wall_d, min_pair)
            obj += val
            # Gradient of min(wall_d, min_pair)
            # This is non-smooth; use subgradient approximation
            # Simplify: gradient towards increasing all distances
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_pair + 1e-6 and d > 1e-8:
                    diff = (positions[i] - positions[j]) / d
                    grad[i] += diff * 0.5
                    grad[j] -= diff * 0.5
            # Wall gradient
            if wall_d <= min_pair:
                # Push away from nearest wall
                if positions[i][0] == wall_d:
                    grad[i][0] += 1.0
                elif h - positions[i][1] == wall_d:
                    grad[i][1] -= 1.0
                elif positions[i][1] == wall_d:
                    grad[i][1] += 1.0
                else:
                    grad[i][0] -= 1.0

        positions += lr * grad
        # Clamp to rectangle interior
        positions[:, 0] = np.clip(positions[:, 0], 0.01, w - 0.01)
        positions[:, 1] = np.clip(positions[:, 1], 0.01, h - 0.01)

        # Occasional LP evaluation
        if it % 200 == 0:
            sum_r, radii = _solve_radii(positions, w, h)
            if sum_r > best_obj:
                best_obj = sum_r
                best_pos = positions.copy()

        # Adaptive learning rate decay
        lr = max(0.0001, lr * 0.999)

        if time.time() > DEADLINE - 5:
            break

    return best_pos, best_obj


def construct():
    start_time = time.time()
    best_sum_r = 0.0
    best_result = None

    # Try several aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.25, 0.6, 2.5, 1.8]
    seeds = [0, 1, 42, 123, 777]

    for ar in aspect_ratios:
        if time.time() > DEADLINE - 10:
            break
        # Perimeter = 2(w+h) = 4 => w+h = 2
        # w/h = ar => w = ar*h => ar*h + h = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)

        for seed in seeds:
            if time.time() > DEADLINE - 10:
                break
            positions = _initial_positions(N, w, h, seed=seed)
            pos_opt, obj = _gradient_descent(positions, w, h, max_iters=3000, lr=0.005, seed=seed)
            sum_r, radii = _solve_radii(pos_opt, w, h)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = np.column_stack([pos_opt, radii])

            # Also try the initial positions directly
            sum_r0, radii0 = _solve_radii(positions, w, h)
            if sum_r0 > best_sum_r:
                best_sum_r = sum_r0
                best_result = np.column_stack([positions, radii0])

    if best_result is None:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final validation
    positions = best_result[:, :2]
    radii = best_result[:, 2]
    # We need to know w,h for validation; store them implicitly
    # The evaluator checks bounding rectangle perimeter <= 4
    # Let's verify our result is valid
    # Find bounding box
    min_x = np.min(positions[:, 0] - radii)
    max_x = np.max(positions[:, 0] + radii)
    min_y = np.min(positions[:, 1] - radii)
    max_y = np.max(positions[:, 1] + radii)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2 * (bw + bh)
    if perim > 4.0 + 1e-6:
        # Scale down
        scale = 4.0 / perim * 0.999
        positions = (positions - np.array([(min_x + max_x)/2, (min_y + max_y)/2])) * scale + np.array([(min_x + max_x)/2, (min_y + max_y)/2])
        radii = radii * scale

    return np.column_stack([positions, radii])
# EVOLVE-BLOCK-END
