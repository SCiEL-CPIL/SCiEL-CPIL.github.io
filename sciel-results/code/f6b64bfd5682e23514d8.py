# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 620.0
_SAFETY_MARGIN = 5.0


def _area_from_positions(x, num_points=20000):
    """
    Compute the area of the union of the 64 triangles using the prescribed quadrature.
    x: array of shape (64,)
    Returns: float area
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)

    w = (1.0 - y) / n

    diffs = np.diff(L_sorted, axis=1)
    capped_diffs = np.minimum(diffs, w[:, None])
    union_len = w + capped_diffs.sum(axis=1)

    area = np.mean(union_len)
    return float(area)


def _keich_baseline():
    """
    Classical Keich construction for n a power of 2.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _area_proxy(x, num_points=2000):
    """
    A faster proxy for the area, using fewer quadrature points.
    Used during optimization for speed.
    """
    return _area_from_positions(x, num_points=num_points)


def _subgradient_and_value(x, num_points=5000):
    """
    Compute the area and its subgradient (numerical) at x.
    """
    n = 64
    h = 1e-5
    area0 = _area_from_positions(x, num_points=num_points)
    grad = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        grad[i] = (_area_from_positions(xp, num_points=num_points) -
                   _area_from_positions(xm, num_points=num_points)) / (2 * h)
    return area0, grad


def _subgradient_descent(x0, max_iter=300, num_points=5000, step_size=0.002):
    """
    Subgradient descent with adaptive step size.
    """
    n = 64
    x = x0.copy()
    current_area = _area_from_positions(x, num_points=num_points)
    step = step_size
    best_x = x.copy()
    best_area = current_area

    for it in range(max_iter):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        area, grad = _subgradient_and_value(x, num_points=num_points)
        if area < best_area:
            best_area = area
            best_x = x.copy()

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        grad = grad / grad_norm
        x_new = x - step * grad
        new_area = _area_from_positions(x_new, num_points=num_points)

        if new_area < current_area:
            x = x_new
            current_area = new_area
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.7
            if step < 1e-9:
                break

    return best_x, best_area


def _coordinate_descent_scan(x, max_passes=20, opt_points=2000):
    """
    Coordinate descent with a 1D scan over each coordinate.
    """
    n = 64
    x = x.copy()
    current_area = _area_proxy(x, num_points=opt_points)

    for _pass in range(max_passes):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break

            x_i_current = x[i]
            search_width = 0.05 / (1 + 0.5 * _pass)
            search_range = np.linspace(x_i_current - search_width, x_i_current + search_width, 50)

            best_xi = x_i_current
            best_local_area = current_area

            for xi in search_range:
                x_temp = x.copy()
                x_temp[i] = xi
                area = _area_proxy(x_temp, num_points=opt_points)
                if area < best_local_area - 1e-12:
                    best_local_area = area
                    best_xi = xi

            if best_xi != x_i_current:
                x[i] = best_xi
                current_area = best_local_area
                improved = True

        if not improved:
            break

    return x


def _soft_min_proxy(x, num_points=2000, beta=50.0):
    """
    Smooth surrogate for the union area using soft-min.
    The union length at height y is w + sum over consecutive sorted pairs of min(diff, w).
    We approximate min(diff, w) via soft-min to get a smooth objective.
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)

    w = (1.0 - y) / n

    diffs = np.diff(L_sorted, axis=1)
    # soft_min(diff, w) ≈ -1/beta * log(sum(exp(-beta*diff), exp(-beta*w)))
    # = -1/beta * log(exp(-beta*diff) + exp(-beta*w))
    # Use log-sum-exp for stability
    log_terms = np.stack([-beta * diffs, -beta * w[:, None]], axis=2)
    max_log = np.max(log_terms, axis=2, keepdims=True)
    log_sum = max_log + np.log(np.sum(np.exp(log_terms - max_log), axis=2))
    soft_min_vals = -log_sum / beta

    union_len = w + soft_min_vals.sum(axis=1)
    area = np.mean(union_len)
    return float(area)


def _scipy_optimize(x0, max_iter=200, num_points=5000):
    """
    Use scipy's L-BFGS-B on the soft-min surrogate, then polish with exact area.
    """
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return x0

    def objective(x):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            return 1e6
        return _soft_min_proxy(x, num_points=num_points, beta=50.0)

    def gradient(x):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            return np.zeros(64)
        h = 1e-5
        grad = np.zeros(64)
        f0 = objective(x)
        for i in range(64):
            xp = x.copy()
            xp[i] += h
            xm = x.copy()
            xm[i] -= h
            grad[i] = (objective(xp) - objective(xm)) / (2 * h)
        return grad

    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                       options={'maxiter': max_iter, 'maxfun': 5000, 'gtol': 1e-5})
        x_opt = res.x
        area_opt = _area_from_positions(x_opt)
        return x_opt
    except Exception:
        return x0


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    # Start with the Keich baseline
    x = _keich_baseline()
    best_x = x.copy()
    best_area = _area_from_positions(x)

    # Phase 1: Coordinate descent with scan (fast, robust)
    x_cd = _coordinate_descent_scan(x, max_passes=30, opt_points=2000)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2: Subgradient descent
    x_subgrad, area_subgrad = _subgradient_descent(best_x, max_iter=200, num_points=5000, step_size=0.002)
    if area_subgrad < best_area:
        best_x = x_subgrad
        best_area = area_subgrad

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3: Scipy L-BFGS-B on soft-min surrogate
    x_scipy = _scipy_optimize(best_x, max_iter=150, num_points=5000)
    area_scipy = _area_from_positions(x_scipy)
    if area_scipy < best_area:
        best_x = x_scipy
        best_area = area_scipy

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 4: Final fine-tuning with coordinate descent on full resolution
    x_final = _coordinate_descent_scan(best_x, max_passes=80, opt_points=10000)
    area_final = _area_from_positions(x_final)
    if area_final < best_area:
        best_x = x_final
        best_area = area_final

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 5: Random restarts + local search
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        x_pert = best_x + rng.normal(0, 0.01, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=10, opt_points=2000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    return best_x


# EVOLVE-BLOCK-END
