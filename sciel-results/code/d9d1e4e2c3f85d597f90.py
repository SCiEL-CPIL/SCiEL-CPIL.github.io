# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses multi-start SLSQP optimization with non-overlap and boundary
    constraints. Returns (centers, sides).
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 570  # safety margin before 630s kill

    def violation(x):
        """Return total constraint violation (0 if feasible)."""
        cx, cy, s = x[:n], x[n:2*n], x[2*n:3*n]
        v = 0.0
        # Boundary: each square inside [-tol, 1+tol]^2
        for i in range(n):
            v += max(0.0, (cx[i] - s[i]/2) - (-tol))  # left edge < -tol
            v += max(0.0, (cx[i] + s[i]/2) - (1.0 + tol))  # right edge > 1+tol
            v += max(0.0, (cy[i] - s[i]/2) - (-tol))  # bottom edge < -tol
            v += max(0.0, (cy[i] + s[i]/2) - (1.0 + tol))  # top edge > 1+tol
        # Non-overlap: for each pair, |dx| >= t or |dy| >= t where t=(si+sj)/2-tol
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # Violation if both dx < t and dy < t
                if dx < t and dy < t:
                    v += (t - dx) + (t - dy)
        return v

    def objective(x):
        """Negative sum of side lengths (we minimize -sum)."""
        s = x[2*n:3*n]
        return -np.sum(s)

    def is_feasible(x):
        return violation(x) < 1e-6

    # Grid baseline as initial guess
    k = 3  # ceil(sqrt(10)) = 4, but 3x4 grid works better
    side = 1.0 / 4.0
    idx = np.arange(n)
    cols = idx % 4
    rows = idx // 4
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    x0 = np.concatenate([cx0, cy0, s0])

    best_x = x0.copy()
    best_obj = objective(x0)

    # Multiple random starts
    rng = np.random.default_rng(42)
    n_starts = 50

    for start in range(n_starts):
        if time.time() > deadline:
            break

        if start == 0:
            x_init = x0.copy()
        else:
            # Random feasible-ish initialization
            cx_init = rng.uniform(0.1, 0.9, n)
            cy_init = rng.uniform(0.1, 0.9, n)
            s_init = rng.uniform(0.05, 0.35, n)
            # Clip to ensure basic feasibility at start
            for i in range(n):
                s_init[i] = min(s_init[i], 0.5)
                cx_init[i] = np.clip(cx_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
                cy_init[i] = np.clip(cy_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
            x_init = np.concatenate([cx_init, cy_init, s_init])

        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-8, 1.0)] * n

        try:
            res = minimize(
                objective, x_init, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            x_res = res.x
            # Enforce minimum side
            x_res[2*n:3*n] = np.maximum(x_res[2*n:3*n], 1e-8)
            obj_res = objective(x_res)

            if is_feasible(x_res) and obj_res < best_obj:
                best_obj = obj_res
                best_x = x_res.copy()
        except Exception:
            continue

    # Extract result
    cx = best_x[:n]
    cy = best_x[n:2*n]
    s = best_x[2*n:3*n]

    # Final cleanup: ensure all sides > 0 and finite
    s = np.maximum(s, 1e-8)
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
