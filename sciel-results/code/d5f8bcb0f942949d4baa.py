# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 625.0
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


def _coordinate_descent_scan(x, max_passes=20, opt_points=2000):
    """
    Coordinate descent with a 1D scan over each coordinate.
    For each coordinate i, we scan a range of values and pick the one that
    minimizes the proxy area. This is more robust than simple +/- step.
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
            
            # Adaptive search range: start wide, narrow down
            # Use a tighter range as we converge
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


def _subgradient_descent(x0, max_iter=200, num_points=5000, step_size=0.001):
    """
    Subgradient descent using the piecewise-linear structure of the area function.
    The area is a convex piecewise-linear function of x, so we can use subgradients.
    """
    n = 64
    x = x0.copy()
    current_area = _area_from_positions(x, num_points=num_points)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Compute subgradient numerically
        grad = np.zeros(n)
        h = 1e-6
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (_area_from_positions(x_plus, num_points=num_points) - 
                      _area_from_positions(x_minus, num_points=num_points)) / (2 * h)
        
        # Normalize gradient to prevent overshooting
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Take step
        x_new = x - step_size * grad
        
        # Check if we improved
        new_area = _area_from_positions(x_new, num_points=num_points)
        if new_area < current_area:
            x = x_new
            current_area = new_area
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return x


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
    # Use a moderate number of points for the proxy
    x_cd = _coordinate_descent_scan(x, max_passes=30, opt_points=2000)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2: Try to further refine with subgradient descent
    # Use a moderate grid for subgradient to balance accuracy and speed
    x_subgrad = _subgradient_descent(best_x, max_iter=100, num_points=5000, step_size=0.001)
    area_subgrad = _area_from_positions(x_subgrad)
    if area_subgrad < best_area:
        best_x = x_subgrad
        best_area = area_subgrad

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3: Final fine-tuning with coordinate descent on full resolution
    # Use more points for better accuracy in the final polish
    x_final = _coordinate_descent_scan(best_x, max_passes=80, opt_points=10000)
    area_final = _area_from_positions(x_final)
    if area_final < best_area:
        best_x = x_final
        best_area = area_final

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 4: Random restarts + local search
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        # Perturb the best solution
        x_pert = best_x + rng.normal(0, 0.01, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=10, opt_points=2000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    return best_x


# EVOLVE-BLOCK-END
