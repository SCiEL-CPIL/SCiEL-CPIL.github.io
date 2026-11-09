# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    import time
    t0 = time.time()
    DEADLINE = t0 + 600.0
    n = 10

    def _objective(params):
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        # side bounds: [1e-4, 0.4]
        s = np.clip(s, 1e-4, 0.4)
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        # containment penalty
        pen = 0.0
        for i in range(n):
            lo = min(cx[i] - s[i]/2.0, cy[i] - s[i]/2.0)
            hi = max(cx[i] + s[i]/2.0, cy[i] + s[i]/2.0)
            if lo < -TOL:
                pen += (lo + TOL) ** 2
            if hi > 1.0 + TOL:
                pen += (hi - 1.0 - TOL) ** 2
        # overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - TOL
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    pen += (t - dx) ** 2 + (t - dy) ** 2
        return -np.sum(s) + 100.0 * pen

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2.0 < -tol or cx[i] + s[i]/2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i]/2.0 < -tol or cy[i] + s[i]/2.0 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    # ---- Analytic 3x4 layout: 3 rows of 4 squares (side 1/4), 1 row of 3 squares (side 1/3)
    # Squares 0..8: 3 rows x 4 cols, side 0.25
    # Squares 9..11? We only have 10. Let's do: rows 0,1,2 have 4 squares each (12 total) - too many.
    # Instead: 3 cols x 4 rows. Bottom 3 rows: 4 squares each? No, 3 cols means 3 per row.
    # Let me reconsider: 4 cols x 3 rows = 12 cells, use 10. But better: 3 cols x 4 rows = 12 cells, use 10.
    # Top row (row 3): 3 squares of side 1/3. Bottom 3 rows: 4 squares each? No, 3 cols.
    # Let's do: 4 cols. Bottom 2 rows: 4 squares of side 1/4. Top 2 rows: need 2 more squares.
    # Actually simplest: 3 cols x 4 rows.
    # Rows 0,1,2: 4 squares each would need 4 cols. Let me use 4 cols x 3 rows for 12, take 10.
    # But the 3x4 with different sizes: 
    # Let's use: 4 squares in bottom row (side 1/4), 4 in middle (side 1/4), 2 in top (side 1/2)? No, 2*0.5=1.0 ok.
    # That's 10 squares: 8 of side 0.25, 2 of side 0.5. Sum = 8*0.25 + 2*0.5 = 2.0+1.0 = 3.0.
    # Better: 3x4 grid. 3 cols, 4 rows. 12 cells, use 10.
    # Top row: 3 squares of side 1/3. Next 3 rows: 4 squares each? No, 3 cols.
    # Let me just do 4 cols x 3 rows = 12 cells, place 10 squares of side 1/4. Sum = 10/4 = 2.5 (baseline).
    # Better analytic: 3 cols x 4 rows. 
    # Row 3 (top): 3 squares, side 1/3. 
    # Rows 0,1,2: 4 squares each? Need 4 cols. Conflict.
    # Let me try: 4 cols. Rows 0,1: 4 squares of side 1/4. Row 2: 2 squares of side 1/2. Total 10. Sum = 8*0.25+2*0.5=3.0.
    # Even better: try 3x4 with 3 cols.
    # Row 3: 3 squares side 1/3. Rows 0,1,2: 3 squares each side 1/3? That's 12, use 10. Sum=10/3=3.33.
    # But can we make some bigger? If we use 3 cols, max side is 1/3 per row if 3 per row.
    # Let me go with the 4-col 2-row + 2-col 1-row: sum 3.0.
    # Or: 5 cols? 5*0.2=1.0. 10 squares of 0.2 = 2.0. Worse.
    # Best analytic: 3x4 grid, 3 cols, 4 rows, 10 squares of side 1/3 = 3.333.
    # Place: rows 0,1,2 full (3 each = 9), row 3 has 1 square. All side 1/3.
    # But row 3 has 1 square of side 1/3, leaving 2/3 empty. Can we make it bigger?
    # If row 3 has 1 square, it could be side 1.0! But then it overlaps with rows below?
    # No, rows are stacked. Row 3 is top. If it's side 1.0, it occupies full width at top.
    # Rows 0,1,2 have 3 squares each of side 1/3. 
    # Sum = 9*(1/3) + 1.0 = 3.0 + 1.0 = 4.0. 
    # But does the top square (side 1.0) overlap with row 2 squares (side 1/3)?
    # Row 2 is at y in [2/3, 1]. Top square at y in [0, 1] if centered at 0.5. Overlap!
    # Need to stack properly. Let me define y-ranges.
    # Let me just use the 3x4 uniform grid for now and let local search improve.
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # 3 cols, 4 rows, 10 squares
    for i in range(n):
        col = i % 3
        row = i // 3
        side = 1.0 / 3.0
        cx = (col + 0.5) * side
        cy = (row + 0.5) * side
        centers[i, 0] = cx
        centers[i, 1] = cy
        sides[i] = side

    # Ensure feasibility
    if not _is_feasible(centers, sides):
        # Fallback to baseline grid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)

    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = np.sum(sides)

    # Local search
    x0 = np.concatenate([sides, centers[:, 0], centers[:, 1]])
    bounds = [(1e-4, 0.4)] * n + [(0.0, 1.0)] * (2 * n)

    rng = np.random.default_rng(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > DEADLINE - 60:
            break
        if r == 0:
            x_start = x0
        else:
            # Perturb
            x_start = x0 + rng.normal(0, 0.05, size=x0.shape)
            x_start = np.clip(x_start, [1e-4] * n + [0.0] * (2 * n), [0.4] * n + [1.0] * (2 * n))

        try:
            res = minimize(_objective, x_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < 0:
                s_opt = np.clip(res.x[:n], 1e-4, 0.4)
                cx_opt = np.clip(res.x[n:2*n], 0.0, 1.0)
                cy_opt = np.clip(res.x[2*n:3*n], 0.0, 1.0)
                c_opt = np.stack([cx_opt, cy_opt], axis=1)
                if _is_feasible(c_opt, s_opt):
                    score = np.sum(s_opt)
                    if score > best_score:
                        best_score = score
                        best_centers = c_opt
                        best_sides = s_opt
        except Exception:
            pass

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
