"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural idea (Delaunay/geometry-based): optimize ONLY the 52 center
coordinates. Radii are derived geometrically from the center layout (largest
feasible radius = min(half nearest-neighbor distance, boundary distance)),
which enforces a tight-packing structure and shrinks the search space.

Refinement machinery (from inspirations): analytical-gradient L-BFGS-B with
multi-start, plus a hard validity-repair pass that guarantees a feasible
result within the 150 s wall-clock budget.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START


def _radii_from_centers(c):
    """Largest feasible radius for each circle given the center layout.

    r_i = min( min_j!=i 0.5*dist(i,j), min(x_i, y_i, 1-x_i, 1-y_i) )
    """
    n = len(c)
    d = np.sqrt((c[:, 0:1] - c[1:, 0:1].T) ** 2 + (c[:, 1:2] - c[1:, 1:2].T) ** 2)
    np.fill_diagonal(d, np.inf)
    r = 0.5 * d.min(axis=1)
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]),
                                 np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1])))
    return np.maximum(r, 0.0)


def _objective_and_grad(c):
    """Objective = -sum(radii) (to minimize), with analytical gradient wrt c.

    Only the nearest neighbor j(i) and the active boundary b(i) constrain r_i.
    d r_i / d c_i = 0.5 * u_ij + v_b ;  d r_i / d c_j = -0.5 * u_ij
    """
    n = len(c)
    d = np.sqrt((c[:, 0:1] - c[1:, 0:1].T) ** 2 + (c[:, 1:2] - c[1:, 1:2].T) ** 2)
    np.fill_diagonal(d, np.inf)
    jn = d.argmin(axis=1)
    r_n = 0.5 * d[np.arange(n), jn]
    db = np.column_stack([c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]])
    bn = db.argmin(axis=1)
    r_b = db[np.arange(n), bn]
    r = np.minimum(r_n, r_b)
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    grad = np.zeros_like(c)
    for i in range(n):
        if r[i] <= 1e-12:
            continue
        j = jn[i]
        dx = c[i, 0] - c[j, 0]
        dy = c[i, 1] - c[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 1e-12:
            u = np.array([dx / dist, dy / dist])
            if r_n[i] <= r_b[i]:            # nearest-neighbor constraint active
                grad[i, 0] += 0.5 * u[0]
                grad[i, 1] += 0.5 * u[1]
                grad[j, 0] -= 0.5 * u[0]
                grad[j, 1] -= 0.5 * u[1]
        if r_b[i] <= r_n[i]:                # boundary constraint active
            v = np.array([1.0 if bn[i] == 0 else -1.0 if bn[i] == 2 else 0.0,
                          1.0 if bn[i] == 1 else -1.0 if bn[i] == 3 else 0.0])
            grad[i, 0] += v[0]
            grad[i, 1] += v[1]
    return obj, grad


def _is_valid(c, r, tol=1e-6):
    n = len(c)
    for i in range(n):
        if (c[i, 0] - r[i] < -tol or c[i, 1] - r[i] < -tol or
                c[i, 0] + r[i] > 1.0 + tol or c[i, 1] + r[i] > 1.0 + tol):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _repair(c, r):
    """Guarantee a feasible packing: recompute r geometrically (valid by
    construction), then shrink any residual violations. Returns (c, r, valid)."""
    n = len(c)
    c = np.clip(c, 0.0, 1.0)
    r = _radii_from_centers(c)
    for _ in range(300):
        bad = False
        for i in range(n):
            mb = min(c[i, 0], c[i, 1], 1.0 - c[i, 0], 1.0 - c[i, 1])
            if r[i] > mb + 1e-9:
                r[i] = max(0.0, mb)
                bad = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                if dist < r[i] + r[j]:
                    excess = (r[i] + r[j] - dist) / 2.0 + 1e-9
                    r[i] = max(0.0, r[i] - excess)
                    r[j] = max(0.0, r[j] - excess)
                    bad = True
        if not bad:
            break
    r = np.maximum(r, 0.0)
    return c, r, _is_valid(c, r)


def _grid_start(n):
    side = int(np.ceil(np.sqrt(n)))
    c = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                  for i in range(side) for j in range(side)][:n], dtype=float)
    return c


def construct():
    start = time.time()
    deadline = start + 120.0          # explicit guard, safety margin under 150 s

    def _evaluate(c):
        c, r, _ = _repair(c, _radii_from_centers(c))
        return c, r, float(np.sum(r))

    c_grid = _grid_start(N)
    best_c, best_r, best_sum = _evaluate(c_grid)

    # Multi-start: perturb the grid layout and optimize only the centers.
    rng = np.random.default_rng(12345)
    it = 0
    while time.time() < deadline - 8.0:
        it += 1
        scale = 0.03 + 0.05 * min(it, 12) / 12.0
        c0 = np.clip(c_grid + rng.uniform(-scale, scale, c_grid.shape), 0.0, 1.0)
        try:
            res = minimize(_objective_and_grad, c0, jac=True, method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * (2 * N),
                           options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
            c_opt = np.clip(res.x, 0.0, 1.0)
        except Exception:
            c_opt = c0
        c_cand, r_cand, s_cand = _evaluate(c_opt)
        if s_cand > best_sum + 1e-9:
            best_c, best_r, best_sum = c_cand, r_cand, s_cand

    # Final guaranteed-valid result.
    final_c, final_r, _ = _repair(best_c, best_r)
    out = np.zeros((N, 3))
    out[:, 0] = final_c[:, 0]
    out[:, 1] = final_c[:, 1]
    out[:, 2] = final_r
    return out
# EVOLVE-BLOCK-END
