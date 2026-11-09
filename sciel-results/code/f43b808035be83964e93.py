# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

TOL = 1e-7
N = 10


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 590.0  # safety margin

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(N):
            if cx[i] - s[i] / 2.0 < -tol or cx[i] + s[i] / 2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i] / 2.0 < -tol or cy[i] + s[i] / 2.0 > 1.0 + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def _objective(params):
        s = params[:N]
        cx = params[N:2 * N]
        cy = params[2 * N:3 * N]
        s = np.clip(s, 1e-4, 0.5)
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        pen = 0.0
        # Containment penalty
        for i in range(N):
            lo_x = cx[i] - s[i] / 2.0
            hi_x = cx[i] + s[i] / 2.0
            lo_y = cy[i] - s[i] / 2.0
            hi_y = cy[i] + s[i] / 2.0
            if lo_x < -TOL:
                pen += (lo_x + TOL) ** 2
            if hi_x > 1.0 + TOL:
                pen += (hi_x - 1.0 - TOL) ** 2
            if lo_y < -TOL:
                pen += (lo_y + TOL) ** 2
            if hi_y > 1.0 + TOL:
                pen += (hi_y - 1.0 - TOL) ** 2
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                t = (s[i] + s[j]) / 2.0 - TOL
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    pen += (t - dx) ** 2 + (t - dy) ** 2
        return -np.sum(s) + 50.0 * pen

    def _make_grid_layout(rows, cols_per_row, total=N):
        """Create a grid layout with specified rows and cols per row."""
        centers = np.zeros((N, 2))
        sides = np.zeros(N)
        idx = 0
        y_offset = 0.0
        for row_idx, ncols in enumerate(rows):
            if idx >= total:
                break
            side = 1.0 / ncols
            # Distribute vertically: each row has height = max side in that row
            row_height = side
            for col in range(ncols):
                if idx >= total:
                    break
                cx = (col + 0.5) * side
                cy = y_offset + row_height / 2.0
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                sides[idx] = side
                idx += 1
            y_offset += row_height
        # Fill remaining squares if any (shouldn't happen with proper rows)
        while idx < N:
            centers[idx, 0] = 0.5
            centers[idx, 1] = y_offset + 0.05
            sides[idx] = 0.01
            idx += 1
        return centers, sides

    def _optimize_from_start(x_start, bounds, maxiter=500):
        """Run L-BFGS-B optimization from a given start point."""
        try:
            res = minimize(_objective, x_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxls': 50})
            return res
        except Exception:
            return None

    # Define bounds
    bounds = [(1e-4, 0.5)] * N + [(0.0, 1.0)] * (2 * N)

    # Generate diverse starting points
    start_points = []

    # 1. Uniform 3x4 grid (3 cols, 4 rows)
    c, s = _make_grid_layout([3, 3, 3, 1])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 2. Uniform 4x3 grid (4 cols, 3 rows) - but only 10 squares
    c, s = _make_grid_layout([4, 4, 2])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 3. 2 rows of 5
    c, s = _make_grid_layout([5, 5])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 4. 5 rows of 2
    c, s = _make_grid_layout([2, 2, 2, 2, 2])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 5. Mixed: row of 4, row of 3, row of 3
    c, s = _make_grid_layout([4, 3, 3])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 6. Mixed: row of 3, row of 4, row of 3
    c, s = _make_grid_layout([3, 4, 3])
    start_points.append(np.concatenate([s, c[:, 0], c[:, 1]]))

    # 7. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        s_rand = np.full(N, 0.25) + rng.normal(0, 0.03, N)
        s_rand = np.clip(s_rand, 0.1, 0.4)
        cx_rand = rng.uniform(0.1, 0.9, N)
        cy_rand = rng.uniform(0.1, 0.9, N)
        x_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        start_points.append(x_rand)

    # Optimize from each start point
    best_score = 0.0
    best_centers = None
    best_sides = None

    # First, check feasibility of analytic layouts
    for c, s in [
        _make_grid_layout([3, 3, 3, 1]),
        _make_grid_layout([4, 4, 2]),
        _make_grid_layout([5, 5]),
        _make_grid_layout([2, 2, 2, 2, 2]),
        _make_grid_layout([4, 3, 3]),
        _make_grid_layout([3, 4, 3]),
    ]:
        if _is_feasible(c, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = c.copy()
                best_sides = s.copy()

    # Now optimize from each start
    for sp_idx, x_start in enumerate(start_points):
        if time.time() > DEADLINE - 30:
            break
        res = _optimize_from_start(x_start, bounds, maxiter=400)
        if res is not None and res.fun < 0:
            s_opt = np.clip(res.x[:N], 1e-4, 0.5)
            cx_opt = np.clip(res.x[N:2 * N], 0.0, 1.0)
            cy_opt = np.clip(res.x[2 * N:3 * N], 0.0, 1.0)
            c_opt = np.stack([cx_opt, cy_opt], axis=1)
            if _is_feasible(c_opt, s_opt):
                score = np.sum(s_opt)
                if score > best_score:
                    best_score = score
                    best_centers = c_opt.copy()
                    best_sides = s_opt.copy()

    # If we still have time, do a second round of refinement on the best solution
    if time.time() < DEADLINE - 60 and best_centers is not None:
        x_best = np.concatenate([best_sides, best_centers[:, 0], best_centers[:, 1]])
        # Try multiple small perturbations around the best
        for perturb_scale in [0.01, 0.02, 0.05]:
            if time.time() > DEADLINE - 30:
                break
            for _ in range(5):
                x_pert = x_best + rng.normal(0, perturb_scale, size=x_best.shape)
                x_pert = np.clip(x_pert, [1e-4] * N + [0.0] * (2 * N),
                                 [0.5] * N + [1.0] * (2 * N))
                res = _optimize_from_start(x_pert, bounds, maxiter=300)
                if res is not None and res.fun < 0:
                    s_opt = np.clip(res.x[:N], 1e-4, 0.5)
                    cx_opt = np.clip(res.x[N:2 * N], 0.0, 1.0)
                    cy_opt = np.clip(res.x[2 * N:3 * N], 0.0, 1.0)
                    c_opt = np.stack([cx_opt, cy_opt], axis=1)
                    if _is_feasible(c_opt, s_opt):
                        score = np.sum(s_opt)
                        if score > best_score:
                            best_score = score
                            best_centers = c_opt.copy()
                            best_sides = s_opt.copy()

    # Final fallback
    if best_centers is None or not _is_feasible(best_centers, best_sides):
        k = 4
        side = 1.0 / k
        idx = np.arange(N)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(N, side, dtype=np.float64)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
