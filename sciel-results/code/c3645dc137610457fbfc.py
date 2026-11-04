"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # safety margin before 120s limit
START_TIME = None


def _is_valid(xyz):
    """Check if circles are disjoint and inside unit square."""
    n = len(xyz)
    # Check boundaries
    for i in range(n):
        x, y, r = xyz[i]
        if r < 1e-10:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = xyz[i]
            xj, yj, rj = xyz[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < min_dist ** 2 - 1e-9:
                return False
    return True


def _objective(params):
    """
    Objective to minimize: -sum(radii) + penalty for constraint violations.
    params: [x0, y0, r0, x1, y1, r1, ...]
    """
    n = N
    xyz = params.reshape(n, 3)
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    rs = xyz[:, 2]

    # Negative sum of radii (we maximize)
    obj = -np.sum(rs)

    # Boundary penalties
    # x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
    bound_viol = 0.0
    bound_viol += np.sum(np.maximum(0, -xs + rs))
    bound_viol += np.sum(np.maximum(0, xs + rs - 1))
    bound_viol += np.sum(np.maximum(0, -ys + rs))
    bound_viol += np.sum(np.maximum(0, ys + rs - 1))

    # Pairwise distance penalties: dist >= ri + rj
    pair_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = rs[i] + rs[j]
            if dist < min_d:
                pair_viol += (min_d - dist)

    # Small radius penalty to avoid degenerate solutions
    small_r_pen = np.sum(np.maximum(0, 0.001 - rs))

    penalty = 1000.0 * (bound_viol + pair_viol) + 100.0 * small_r_pen
    return obj + penalty


def _objective_vectorized(params):
    """Vectorized objective for speed."""
    n = N
    xyz = params.reshape(n, 3)
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    rs = xyz[:, 2]

    obj = -np.sum(rs)

    # Boundary violations
    bv = (np.sum(np.maximum(0, -xs + rs)) +
          np.sum(np.maximum(0, xs + rs - 1)) +
          np.sum(np.maximum(0, -ys + rs)) +
          np.sum(np.maximum(0, ys + rs - 1)))

    # Pairwise violations using broadcasting
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_d = rs[:, None] + rs[None, :]
    # Only consider upper triangle
    iu = np.triu_indices(n, k=1)
    pair_viol = np.sum(np.maximum(0, min_d[iu] - dist[iu]))

    small_r_pen = np.sum(np.maximum(0, 0.001 - rs))

    penalty = 1000.0 * (bv + pair_viol) + 100.0 * small_r_pen
    return obj + penalty


def _hexagonal_initial():
    """Create a hexagonal packing initial configuration."""
    # Hexagonal packing: rows offset
    # For 26 circles, try a 5x6 or 6x5 grid with hex offset
    # Let's use a 6x5 arrangement with hexagonal offset
    rows = 6
    cols = 5
    # Total = 30, we take first 26
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N:
                break
            # Hex offset: odd rows shifted
            x = (j + 0.5) / cols + (0.0 if i % 2 == 0 else 0.5 / cols)
            y = (i + 0.5) / rows
            # Radius based on spacing
            # In hex packing, horizontal spacing = 2r, vertical = sqrt(3)*r
            # But we're in a unit square, so let's estimate
            r = 0.5 / cols * 0.95
            circles.append([x, y, r])
    return np.array(circles[:N])


def _grid_initial():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            cells.append([x, y, r])
    return np.array(cells[:N])


def _random_initial(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N)
    ys = rng.uniform(0.1, 0.9, N)
    rs = rng.uniform(0.02, 0.1, N)
    return np.column_stack([xs, ys, rs])


def _optimize_from_initial(xyz_init, maxiter=2000, tol=1e-10):
    """Run L-BFGS-B optimization from initial point."""
    params = xyz_init.flatten()

    def obj(p):
        return _objective_vectorized(p)

    result = minimize(obj, params, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': 1e-8})
    return result


def _polish(xyz):
    """Final high-precision polish pass."""
    params = xyz.flatten()

    def obj(p):
        return _objective_vectorized(p)

    result = minimize(obj, params, method='L-BFGS-B',
                      options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14})
    return result.x.reshape(N, 3)


def construct():
    global START_TIME
    START_TIME = time.time()

    best_xyz = None
    best_score = -np.inf

    # Try multiple initializations
    inits = []

    # Hexagonal
    inits.append(_hexagonal_initial())

    # Grid
    inits.append(_grid_initial())

    # Random initializations with different seeds
    for seed in range(9):
        inits.append(_random_initial(seed * 42 + 7))

    # Also try some structured randoms
    # Concentric-ish: larger circles near center
    rng = np.random.RandomState(123)
    for _ in range(3):
        xs = rng.uniform(0.2, 0.8, N)
        ys = rng.uniform(0.2, 0.8, N)
        # Make some circles bigger
        rs = rng.uniform(0.03, 0.12, N)
        inits.append(np.column_stack([xs, ys, rs]))

    for idx, xyz_init in enumerate(inits):
        if time.time() - START_TIME > DEADLINE - 10:
            break

        try:
            result = _optimize_from_initial(xyz_init, maxiter=1500)
            xyz_opt = result.x.reshape(N, 3)

            # Check validity
            if _is_valid(xyz_opt):
                score = np.sum(xyz_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_opt.copy()
        except Exception:
            continue

    # Polish the best solution found
    if best_xyz is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = _polish(best_xyz)
            if _is_valid(polished):
                score = np.sum(polished[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = polished.copy()
        except Exception:
            pass

    if best_xyz is not None:
        return best_xyz

    # Fallback: return a valid grid configuration
    return _grid_initial()


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
