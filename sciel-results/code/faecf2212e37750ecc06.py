# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _union_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(20000) + 0.5) / 20000  # 20000 heights
    # L_i(y) = x_i + a_i * y, w(y) = (1-y)/64
    # Shape: (20000, 64)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (20000,)
    # Sort L along axis=1
    L_sorted = np.sort(L, axis=1)
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (20000, 63)
    # Union length = w + sum(min(diffs, w))
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    return np.mean(union_len)


def _union_area_grad(x):
    """Compute union area and its gradient w.r.t. x."""
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n

    # Sort L and track which original index each sorted position came from
    order = np.argsort(L, axis=1)  # (20000, 64)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)  # (20000, 63)

    # Which pairs are "active" (diff < w)?
    active = (diffs < w[:, np.newaxis]).astype(np.float64)  # (20000, 63)

    # Union length
    union_len = w + np.sum(active * diffs, axis=1)
    area = np.mean(union_len)

    # Gradient: d(area)/d(x_i)
    M = 20000
    grad = np.zeros(n, dtype=np.float64)

    # Vectorized accumulation
    left_idx = order[:, :-1]   # (M, 63)
    right_idx = order[:, 1:]   # (M, 63)

    weight = active / M  # (M, 63)

    r_flat = right_idx.ravel()  # (M*63,)
    l_flat = left_idx.ravel()
    w_flat = weight.ravel()
    a_r = a[r_flat]  # a values at right indices
    a_l = a[l_flat]  # a values at left indices

    np.add.at(grad, r_flat, w_flat * a_r)
    np.add.at(grad, l_flat, -w_flat * a_l)

    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Uses a combination of Keich construction, Differential Evolution, and L-BFGS-B.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_keich = np.array(positions, dtype=np.float64)

    t_start = time.time()
    deadline = t_start + 620.0  # Safety margin

    best_x = x_keich.copy()
    best_area = _union_area(best_x)

    def obj_func(x):
        return _union_area(x)

    def jac_func(x):
        return _union_area_grad(x)[1]

    # 1. Local optimization from Keich point
    try:
        res = minimize(obj_func, x_keich, jac=jac_func, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 2000, 'gtol': 1e-8})
        if res.success and np.all(np.isfinite(res.x)):
            area = _union_area(res.x)
            if area < best_area:
                best_area = area
                best_x = res.x.copy()
    except Exception:
        pass

    # 2. Differential Evolution for global search
    # Use a small population and few iterations to stay within time limits
    # Bounds: The relative positions are what matter. We can bound x to [-1, 1]
    # The triangles have width 1/64. Shifting all x by a constant doesn't change area.
    # So we can fix x_0 = 0 and optimize x_1...x_63? 
    # But DE requires a vector. Let's just bound all to [-1, 1].
    bounds = [(-1.0, 1.0) for _ in range(n)]
    
    try:
        # DE is stochastic. Set seed for reproducibility if needed, but random is fine.
        res_de = differential_evolution(
            obj_func, bounds, 
            maxiter=20, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            polish=False  # We will polish manually
        )
        if res_de.success and np.all(np.isfinite(res_de.x)):
            area_de = _union_area(res_de.x)
            if area_de < best_area:
                best_area = area_de
                best_x = res_de.x.copy()
    except Exception:
        pass

    # 3. Polish the best result found so far with L-BFGS-B
    if time.time() < deadline:
        try:
            res_polish = minimize(obj_func, best_x, jac=jac_func, method='L-BFGS-B',
                                  options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-10})
            if np.all(np.isfinite(res_polish.x)):
                area_polish = _union_area(res_polish.x)
                if area_polish < best_area:
                    best_area = area_polish
                    best_x = res_polish.x.copy()
        except Exception:
            pass

    # 4. Random restarts with L-BFGS-B if time permits
    # Generate random perturbations around the best solution
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        # Perturb the best solution
        # Scale of perturbation: start small, maybe increase if stuck?
        # For now, just small noise.
        noise = rng.normal(0, 0.01, size=n)
        x_start = best_x + noise
        
        try:
            res_rr = minimize(obj_func, x_start, jac=jac_func, method='L-BFGS-B',
                              options={'maxiter': 100, 'maxfun': 1000, 'gtol': 1e-8})
            if np.all(np.isfinite(res_rr.x)):
                area_rr = _union_area(res_rr.x)
                if area_rr < best_area:
                    best_area = area_rr
                    best_x = res_rr.x.copy()
        except Exception:
            break

    # Final check
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich

    return best_x


# EVOLVE-BLOCK-END
