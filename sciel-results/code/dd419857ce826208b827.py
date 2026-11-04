"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _smooth_obj_and_grad(params, n, beta):
    """
    Differentiable objective: maximize sum(r) with soft constraints.
    Returns (objective_value, gradient) where we MINIMIZE -sum(r) + penalty.
    Soft wall penalty: for each circle, violation = max(0, r - dist_to_wall).
    Soft overlap: for each pair, violation = max(0, r_i + r_j - dist).
    Use smooth approx: softplus-like via sqrt(v^2 + eps) - eps (C1 smooth).
    """
    eps = 1e-4
    xy = params[0::3].reshape(n, 2)
    r = params[1::3].reshape(n)
    r = np.maximum(r, 1e-4)

    # Wall distances
    wall = np.minimum(np.minimum(xy[:, 0], 1.0 - xy[:, 0]),
                      np.minimum(xy[:, 1], 1.0 - xy[:, 1]))
    wall_viol = np.maximum(0.0, r - wall)
    # Smooth: sqrt(v^2 + eps) - sqrt(eps)
    wall_smooth = np.sqrt(wall_viol ** 2 + eps) - np.sqrt(eps)

    # Pairwise
    overlap_pen = 0.0
    grad_x = np.zeros_like(xy)
    grad_y = np.zeros_like(xy)
    grad_r = np.zeros(n)

    # Wall gradient
    # d/dx of (r - min(x, 1-x, y, 1-y)) when positive
    for i in range(n):
        if wall_viol[i] > 0:
            v = wall_viol[i]
            # Which wall is closest?
            d_left = xy[i, 0]
            d_right = 1.0 - xy[i, 0]
            d_bot = xy[i, 1]
            d_top = 1.0 - xy[i, 1]
            min_d = min(d_left, d_right, d_bot, d_top)
            # Gradient of wall distance w.r.t. x,y
            gx = 0.0
            gy = 0.0
            if min_d == d_left:
                gx = -1.0
            elif min_d == d_right:
                gx = 1.0
            elif min_d == d_bot:
                gy = -1.0
            elif min_d == d_top:
                gy = 1.0
            # d/dx of smooth(v) = v / sqrt(v^2+eps) * (-gx)  [since v = r - wall, d(v)/dx = -gx]
            factor = v / np.sqrt(v ** 2 + eps)
            grad_x[i, 0] += -factor * gx
            grad_y[i, 0] += -factor * gy
            grad_r[i] += factor

    # Pairwise smooth overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-12)
            min_dist = r[i] + r[j]
            viol = min_dist - dist
            if viol > 0:
                v = viol
                smooth = np.sqrt(v ** 2 + eps) - np.sqrt(eps)
                overlap_pen += smooth
                factor = v / np.sqrt(v ** 2 + eps)
                # d(dist)/dx_i = dx/dist, so d(viol)/dx_i = -dx/dist
                # d(smooth)/dx_i = factor * (-dx/dist)
                grad_x[i, 0] += -factor * (dx / dist)
                grad_y[i, 0] += -factor * (dy / dist)
                grad_x[j, 0] += factor * (dx / dist)
                grad_y[j, 0] += factor * (dy / dist)
                # d(viol)/dr_i = 1, so d(smooth)/dr_i = factor
                grad_r[i] += factor
                grad_r[j] += factor

    objective = -np.sum(r) + beta * (np.sum(wall_smooth) + overlap_pen)

    # Gradient of objective
    grad = np.zeros(3 * n)
    grad[0::3] = grad_x
    grad[1::3] = grad_y
    grad[2::3] = grad_r  # note: -1 from -sum(r) already? No, we added -sum(r) to objective
    # Actually objective = -sum(r) + beta*penalties
    # d(-sum(r))/dr_i = -1
    grad[2::3] -= 1.0

    return objective, grad


def _largest_empty_circle_refine(xy, r, deadline, n_sweeps=15):
    """
    For each circle, compute the maximum radius it can have at its current position
    given all other circles, then grow it. Repeat.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    for sweep in range(n_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Max radius at position i given others
            x, y = xy[i, 0], xy[i, 1]
            wall_r = min(x, 1.0 - x, y, 1.0 - y)
            max_r = wall_r
            for j in range(n):
                if j == i:
                    continue
                dx = x - xy[j, 0]
                dy = y - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                avail = dist - r[j]
                if avail < max_r:
                    max_r = avail
            max_r = max(max_r, 0.001)
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.default_rng(seed)
    xy = rng.rand(n, 2)
    r = np.full(n, 0.01)
    return xy, r


def _gradient_optimize(xy, r, deadline, max_iter=300):
    """
    Use L-BFGS-B with analytic gradients to optimize the packing.
    """
    n = len(xy)

    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.45))

    best_params = params.copy()
    best_total = 0.0
    best_valid = False

    init_total, init_valid = _eval_packing(xy, r)
    if init_valid:
        best_total = init_total
        best_valid = True

    # Adaptive beta: start high, reduce over iterations
    def _objective(params):
        val, _ = _smooth_obj_and_grad(params, n, beta)
        return val

    def _gradient(params):
        _, g = _smooth_obj_and_grad(params, n, beta)
        return g

    # We need beta to be a closure variable; use a list for mutability
    beta_holder = [1000.0]

    def _obj(params):
        val, _ = _smooth_obj_and_grad(params, n, beta_holder[0])
        return val

    def _grad(params):
        _, g = _smooth_obj_and_grad(params, n, beta_holder[0])
        return g

    try:
        result = minimize(
            _obj,
            params,
            jac=_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-8}
        )

        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(n, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)

        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()

    except Exception:
        pass

    # Second pass with lower beta for finer optimization
    if time.time() < deadline - 5.0 and best_valid:
        beta_holder[0] = 50.0
        try:
            result2 = minimize(
                _obj,
                best_params,
                jac=_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-10}
            )
            opt_params2 = result2.x
            opt_xy2 = opt_params2[0::3].reshape(n, 2)
            opt_r2 = np.maximum(opt_params2[1::3], 0.001)

            total2, valid2 = _eval_packing(opt_xy2, opt_r2)
            if valid2 and total2 > best_total:
                best_total = total2
                best_valid = True
                best_params = opt_params2.copy()

        except Exception:
            pass

    final_xy = best_params[0::3].reshape(n, 2)
    final_r = np.maximum(best_params[1::3], 0.001)

    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Multiple diverse initializations
    inits = []

    # Grid inits
    for side in [4, 5, 6]:
        grid = _grid_init(side)
        xy = grid[:, :2].copy()
        r = grid[:, 2].copy()
        inits.append((xy, r))

    # Random inits
    for seed in [1, 2, 3, 4, 5]:
        xy, r = _random_init(N, seed)
        inits.append((xy, r))

    # Process each init
    for idx, (xy, r) in enumerate(inits):
        if time.time() > deadline - 20.0:
            break

        # Step 1: Gradient optimization
        if time.time() < deadline - 10.0:
            g_xy, g_r, g_total, g_valid = _gradient_optimize(
                xy, r, deadline - 10.0, max_iter=200
            )
            if g_valid and g_total > best_total:
                best_total = g_total
                best_valid = True
                best_xy = g_xy
                best_r = g_r

        # Step 2: Largest empty circle refinement
        if time.time() < deadline - 5.0 and best_valid:
            refine_xy, refine_r, refine_total, refine_valid = _largest_empty_circle_refine(
                best_xy, best_r, deadline - 5.0, n_sweeps=10
            )
            if refine_valid and refine_total > best_total:
                best_total = refine_total
                best_valid = True
                best_xy = refine_xy
                best_r = refine_r

    # If no valid result found, fall back to a simple grid
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        grid = _grid_init(side)
        xy = grid[:, :2].copy()
        r = grid[:, 2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
