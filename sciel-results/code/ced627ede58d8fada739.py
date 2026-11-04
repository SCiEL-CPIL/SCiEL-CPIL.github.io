"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
_DEADLINE = 120.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _smooth_max(x, beta):
    """Smooth max of array x using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.max(x)
    if m == 0 and np.all(x == 0):
        return 0.0
    s = np.sum(np.exp(beta * (x - m)))
    return m + np.log(s) / beta


def _compute_radius(pos, w):
    """Compute max feasible radius given positions and width w."""
    h = 2.0 - w
    n = pos.shape[0]
    # Wall constraints: r <= min(x_i, w - x_i, y_i, h - y_i)
    wall_min = np.min(np.minimum(
        np.minimum(pos[:, 0], w - pos[:, 0]),
        np.minimum(pos[:, 1], h - pos[:, 1])
    ))
    # Circle-circle: r <= 0.5 * min pairwise distance
    if n > 1:
        dx = pos[:, 0:1] - pos[0:1, 0]
        dy = pos[:, 1:2] - pos[0:1, 1]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        cc_min = 0.5 * np.min(dists)
    else:
        cc_min = wall_min
    return max(0.0, min(wall_min, cc_min))


def _objective(params, w):
    """Objective: -sum(radii) with smooth constraint handling.
    params: (2N,) array of positions. w: width."""
    pos = params.reshape(N, 2)
    h = 2.0 - w
    # Soft constraint: penalize if circles go outside or overlap
    # Use smooth max of violations
    n = N
    # Wall violations (should be >= 0, penalize negatives)
    wall_viol = np.concatenate([
        -pos[:, 0], w - pos[:, 0],
        -pos[:, 1], h - pos[:, 1]
    ])
    # Circle-circle violations
    if n > 1:
        dx = pos[:, 0:1] - pos[0:1, 0]
        dy = pos[:, 1:2] - pos[0:1, 1]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        # For radius r, need dist >= 2r. We'll compute r and check.
        # Actually, let's just compute the raw radius and use it.
        pass

    r = _compute_radius(pos, w)
    # Penalty for infeasibility (circles outside bounds)
    # If any circle is outside, r should be 0 or negative
    if r <= 0:
        return 100.0  # large penalty
    return -r  # we maximize r (sum = N*r for equal radii)


def _construct_grid_init(w):
    """Generate initial positions on a hexagonal grid inside the rectangle."""
    h = 2.0 - w
    # Try to fit N circles in a hexagonal pattern
    # For a hex grid with spacing s, we need to figure out rows/cols
    # Let's try different arrangements
    best_pos = None
    best_score = -1

    # Try rectangular grids
    for rows in range(1, N + 1):
        cols = (N + rows - 1) // rows
        if cols * rows < N:
            continue
        # Spacing
        sx = w / (cols + 0.5)
        sy = h / (rows + 0.5)
        s = min(sx, sy)
        # Place circles
        pos = []
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N:
                    break
                x = (j + 1) * sx + (0.5 * sx if i % 2 == 1 else 0)
                y = (i + 1) * sy
                x = np.clip(x, 0.01, w - 0.01)
                y = np.clip(y, 0.01, h - 0.01)
                pos.append([x, y])
                count += 1
            if count >= N:
                break
        if len(pos) < N:
            continue
        pos = np.array(pos[:N])
        r = _compute_radius(pos, w)
        score = r * N
        if score > best_score:
            best_score = score
            best_pos = pos

    if best_pos is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        pos = []
        for i in range(side):
            for j in range(side):
                if len(pos) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                pos.append([x, y])
        best_pos = np.array(pos[:N])

    return best_pos


def _nelder_mead_optimize(pos_init, w_init, max_iter=500):
    """Optimize positions and width using Nelder-Mead."""
    def obj(params):
        if _elapsed() > _DEADLINE - 10:
            return 1e10
        w = params[0]
        pos = params[1:].reshape(N, 2)
        h = 2.0 - w
        # Check bounds
        if w <= 0.01 or w >= 1.99:
            return 1e10
        if np.any(pos[:, 0] < -0.1) or np.any(pos[:, 0] > w + 0.1):
            return 1e10
        if np.any(pos[:, 1] < -0.1) or np.any(pos[:, 1] > h + 0.1):
            return 1e10
        r = _compute_radius(pos, w)
        if r <= 0:
            return 1e10
        return -r * N  # minimize negative sum of radii

    x0 = np.concatenate([[w_init], pos_init.flatten()])
    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    if result.success or result.fun < 1e9:
        w_opt = result.x[0]
        pos_opt = result.x[1:].reshape(N, 2)
        return w_opt, pos_opt
    return w_init, pos_init


def _polish(w, pos, iterations=1000):
    """Polish positions using gradient-based optimization with smooth constraints."""
    h = 2.0 - w
    n = N

    # Compute initial radius
    r = _compute_radius(pos, w)
    if r <= 0:
        return w, pos, 0.0

    # Use scipy minimize with BFGS or L-BFGS-B on a smooth objective
    # Objective: maximize r subject to constraints, or equivalently
    # minimize -r with smooth penalty for constraint violations

    def smooth_obj(params):
        if _elapsed() > _DEADLINE - 5:
            return 1e10
        w_p = params[0]
        pos_p = params[1:].reshape(N, 2)
        h_p = 2.0 - w_p
        if w_p <= 0.01 or w_p >= 1.99:
            return 1e10

        # Compute all constraints
        # Wall: x_i >= r, w - x_i >= r, y_i >= r, h - y_i >= r
        # Circle-circle: ||p_i - p_j|| >= 2r

        # For smoothness, use log-sum-exp on the minimum of constraints
        # Actually, let's just compute r and add smooth penalty

        wall_min = np.min(np.minimum(
            np.minimum(pos_p[:, 0], w_p - pos_p[:, 0]),
            np.minimum(pos_p[:, 1], h_p - pos_p[:, 1])
        ))
        if n > 1:
            dx = pos_p[:, 0:1] - pos_p[0:1, 0]
            dy = pos_p[:, 1:2] - pos_p[0:1, 1]
            dists = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dists, np.inf)
            cc_min = 0.5 * np.min(dists)
        else:
            cc_min = wall_min

        r_p = min(wall_min, cc_min)
        if r_p <= 0:
            return 1e10
        return -r_p * N

    x0 = np.concatenate([[w], pos.flatten()])
    try:
        result = minimize(smooth_obj, x0, method='L-BFGS-B',
                          options={'maxiter': iterations, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < 1e9:
            w_opt = result.x[0]
            pos_opt = result.x[1:].reshape(N, 2)
            r_opt = _compute_radius(pos_opt, w_opt)
            return w_opt, pos_opt, r_opt * N
    except Exception:
        pass
    return w, pos, r * N


def construct():
    if _elapsed() > _DEADLINE - 5:
        # Emergency fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    best_score = 0.0
    best_result = None

    # Try multiple initial aspect ratios
    w_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for w_init in w_candidates:
        if _elapsed() > _DEADLINE - 15:
            break

        # Generate initial positions
        pos_init = _construct_grid_init(w_init)
        r_init = _compute_radius(pos_init, w_init)
        score_init = r_init * N

        if score_init > best_score:
            best_score = score_init
            best_result = (w_init, pos_init.copy())

        # Nelder-Mead optimization
        if _elapsed() > _DEADLINE - 10:
            break
        w_nm, pos_nm = _nelder_mead_optimize(pos_init, w_init, max_iter=300)
        r_nm = _compute_radius(pos_nm, w_nm)
        score_nm = r_nm * N
        if score_nm > best_score:
            best_score = score_nm
            best_result = (w_nm, pos_nm.copy())

    # Final polish on best result
    if best_result is not None and _elapsed() < _DEADLINE - 5:
        w_best, pos_best = best_result
        w_pol, pos_pol, score_pol = _polish(w_best, pos_best, iterations=500)
        if score_pol > best_score:
            best_score = score_pol
            best_result = (w_pol, pos_pol)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    w_final, pos_final = best_result
    r_final = _compute_radius(pos_final, w_final)

    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = pos_final[:, 0]
    result[:, 1] = pos_final[:, 1]
    result[:, 2] = r_final

    return result
