"""Circle packing in perimeter-4 rectangle: contact-graph optimization + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    # Precompute upper triangle indices for pairwise constraints
    iu = np.triu_indices(N, k=1)

    def is_valid(x, y, r, w, h, tol=1e-9):
        """Check feasibility with vectorized pairwise checks."""
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(r <= 0):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False
        return True

    def max_radius_for_circle(i, x, y, r, w, h):
        """Compute maximum radius for circle i given current positions and other radii."""
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def expand_radii(x, y, r, w, h, max_iters=300):
        """Greedily expand each circle's radius to its maximum possible value."""
        r = r.copy().astype(float)
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(N):
                r_max = max_radius_for_circle(i, x, y, r, w, h)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r

    def expansion_polish(x, y, r, w, h, time_budget=8.0):
        """
        Detailed expansion polish: try moving each circle to a better position
        to allow larger radius, then expand. Uses multi-directional position search.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        steps = [0.06, 0.04, 0.025, 0.015, 0.008, 0.004, 0.002, 0.001]

        def try_move_circle(i, current_x, current_y, current_r, x_arr, y_arr, r_arr):
            """Try to find a better position for circle i that allows larger radius."""
            best_x, best_y, best_rmax = current_x, current_y, max_radius_for_circle(i, current_x, current_y, r_arr, w, h)
            for angle in angles:
                for step in steps:
                    nx = current_x + step * np.cos(angle)
                    ny = current_y + step * np.sin(angle)
                    # Check wall validity
                    if nx < current_r or nx > w - current_r or ny < current_r or ny > h - current_r:
                        continue
                    # Check pairwise validity
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = nx - x_arr[j]
                        dy = ny - y_arr[j]
                        d2 = dx * dx + dy * dy
                        rs = current_r + r_arr[j]
                        if d2 < rs * rs - 1e-10:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Compute max radius at new position
                    r_wall_new = min(nx, w - nx, ny, h - ny)
                    r_max_new = r_wall_new
                    for j in range(N):
                        if i == j:
                            continue
                        dx = nx - x_arr[j]
                        dy = ny - y_arr[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r_arr[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)
                    if r_max_new > best_rmax + 1e-9:
                        best_x, best_y, best_rmax = nx, ny, r_max_new
            return best_x, best_y, best_rmax

        for outer in range(80):
            if time.time() - start > time_budget:
                break
            improved = False
            # Process circles in order of increasing radius (smaller ones first, more flexible)
            order = np.argsort(r)
            for i in order:
                if time.time() - start > time_budget:
                    break
                # First try direct expansion
                r_max = max_radius_for_circle(i, x, y, r, w, h)
                if r_max > r[i] + 1e-9:
                    # Binary search for exact max
                    lo, hi = r[i], r_max
                    for _ in range(14):
                        mid = (lo + hi) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid
                        if is_valid(x, y, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-10:
                        r[i] = lo
                        improved = True
                        continue

                # Try moving to a better position
                bx, by, brmax = try_move_circle(i, x[i], y[i], r[i], x, y, r)
                if brmax > r[i] + 1e-8:
                    x[i], y[i] = bx, by
                    lo, hi = r[i], brmax
                    for _ in range(14):
                        mid = (lo + hi) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid
                        if is_valid(x, y, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-10:
                        r[i] = lo
                    improved = True
            if not improved:
                break
        return x, y, r

    def penalty_objective(zv, w, h):
        """
        Penalty-based objective for contact-graph optimization.
        z = [x_0..x_N, y_0..y_N, r_0..r_N]
        """
        x = zv[0:N]
        y = zv[N:2*N]
        r = zv[2*N:3*N]

        if w < 0.1 or h < 0.1:
            return 1e6

        obj = -np.sum(r)

        # Wall violations
        wall_viol = (np.maximum(r - x, 0) + np.maximum(x + r - w, 0) +
                     np.maximum(r - y, 0) + np.maximum(y + r - h, 0))
        obj += 50.0 * np.sum(wall_viol)

        # Circle-circle violations (vectorized)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum * r_sum - d2, 0)
        obj += 50.0 * np.sum(overlap)

        # Radius non-negativity
        obj += 50.0 * np.sum(np.maximum(-r, 0))

        return obj

    def slsqp_refine(x, y, r, w, h):
        """SLSQP refinement to tighten the packing."""
        v0 = np.concatenate([x, y, r])
        bounds = []
        for _ in range(N):
            bounds.append((0.0, w))
        for _ in range(N):
            bounds.append((0.0, h))
        for _ in range(N):
            bounds.append((0.0, 0.5 * min(w, h)))

        try:
            result = minimize(
                lambda v: penalty_objective(v, w, h),
                v0, method='SLSQP', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12}
            )
            x_r = result.x[0:N]
            y_r = result.x[N:2*N]
            r_r = result.x[2*N:3*N]
            r_r = np.clip(r_r, 0, 0.5 * min(w, h))
            x_r = np.clip(x_r, 0, w)
            y_r = np.clip(y_r, 0, h)
            r_r = expand_radii(x_r, y_r, r_r, w, h, max_iters=100)
            if is_valid(x_r, y_r, r_r, w, h):
                return x_r, y_r, r_r
        except Exception:
            pass
        return x, y, r

    def contact_graph_optimize(x, y, r, w, h, time_budget=20.0):
        """
        Contact-graph optimization: treat tangencies as constraints and solve
        for positions/radii that maximize sum of radii. Uses L-BFGS-B with
        a penalty formulation that strongly encourages tangency.
        """
        start = time.time()
        v0 = np.concatenate([x, y, r])

        # Build a contact graph: circles that are close together
        # For now, use all pairs as potential contacts
        def contact_objective(zv):
            if time.time() - start > time_budget:
                return 1e6
            xx = zv[0:N]
            yy = zv[N:2*N]
            rr = zv[2*N:3*N]
            if w < 0.1 or h < 0.1:
                return 1e6
            obj = -np.sum(rr)
            # Strong penalty for overlaps
            dx = xx[iu[0]] - xx[iu[1]]
            dy = yy[iu[0]] - yy[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = rr[iu[0]] + rr[iu[1]]
            overlap = np.maximum(r_sum * r_sum - d2, 0)
            obj += 100.0 * np.sum(overlap)
            # Wall constraints
            wall_viol = (np.maximum(rr - xx, 0) + np.maximum(xx + rr - w, 0) +
                         np.maximum(rr - yy, 0) + np.maximum(yy + rr - h, 0))
            obj += 100.0 * np.sum(wall_viol)
            # Encourage tangency: penalize gaps between close circles
            # This is the key structural difference from pure penalty
            d = np.sqrt(np.maximum(d2, 1e-12))
            gap = np.maximum(r_sum - d - 0.001, 0)  # small tolerance
            # Only penalize if they were close (potential contacts)
            close_mask = d < r_sum + 0.05
            obj += 5.0 * np.sum(gap[close_mask])
            return obj

        bounds = []
        for _ in range(N):
            bounds.append((0.0, w))
        for _ in range(N):
            bounds.append((0.0, h))
        for _ in range(N):
            bounds.append((0.0, 0.5 * min(w, h)))

        try:
            result = minimize(
                contact_objective, v0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_r = result.x[0:N]
            y_r = result.x[N:2*N]
            r_r = result.x[2*N:3*N]
            r_r = np.clip(r_r, 0, 0.5 * min(w, h))
            x_r = np.clip(x_r, 0, w)
            y_r = np.clip(y_r, 0, h)
            r_r = expand_radii(x_r, y_r, r_r, w, h, max_iters=100)
            if is_valid(x_r, y_r, r_r, w, h):
                return x_r, y_r, r_r
        except Exception:
            pass
        return x, y, r

    def local_search(x, y, r, w, h, max_iters=150):
        """Local search with random perturbations and re-expansion."""
        best_r_sum = np.sum(r)
        best_state = (x.copy(), y.copy(), r.copy())
        rng = np.random.RandomState(42)

        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            i = rng.randint(N)
            dx = rng.normal(0, 0.015)
            dy = rng.normal(0, 0.015)
            new_x = x.copy()
            new_y = y.copy()
            new_x[i] = np.clip(x[i] + dx, r[i], w - r[i])
            new_y[i] = np.clip(y[i] + dy, r[i], h - r[i])

            new_r = expand_radii(new_x, new_y, r.copy(), w, h, max_iters=80)
            if not is_valid(new_x, new_y, new_r, w, h):
                continue
            new_sum = np.sum(new_r)
            if new_sum > best_r_sum + 1e-9:
                best_r_sum = new_sum
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
                x, y, r = best_state
        return best_state

    # Main optimization loop over multiple rectangle aspect ratios
    np.random.seed(42)
    best_value = -np.inf
    best_circles = None

    # Try different aspect ratios (w + h = 2, so perimeter = 4)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.55, 0.6, 1.7, 0.55, 1.85]

    for w in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue

        # Generate diverse initial packings for this aspect ratio
        initializations = []

        # 1. Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        grid_x = np.array([((i % side) + 0.5) * w / side for i in range(N)])
        grid_y = np.array([((i // side) + 0.5) * h / side for i in range(N)])
        grid_r = np.full(N, 0.5 * min(w, h) / side - 0.01)
        initializations.append((grid_x.copy(), grid_y.copy(), grid_r.copy()))

        # 2. Perturbed grids
        for k in range(3):
            if time.time() > deadline - 30:
                break
            px = grid_x + np.random.randn(N) * 0.02 * min(w, h)
            py = grid_y + np.random.randn(N) * 0.02 * min(w, h)
            pr = grid_r + np.random.randn(N) * 0.005
            px = np.clip(px, 0.01, w - 0.01)
            py = np.clip(py, 0.01, h - 0.01)
            pr = np.clip(pr, 0.005, 0.4 * min(w, h))
            initializations.append((px, py, pr))

        # 3. Random scattered
        for k in range(2):
            if time.time() > deadline - 30:
                break
            rx = np.random.uniform(0.05, w - 0.05, N)
            ry = np.random.uniform(0.05, h - 0.05, N)
            rr = np.full(N, 0.03)
            initializations.append((rx, ry, rr))

        # Process each initialization
        for x, y, r in initializations:
            if time.time() > deadline - 25:
                break

            # Step 1: Expand radii
            r = expand_radii(x, y, r, w, h, max_iters=100)

            # Step 2: Contact graph optimization
            x, y, r = contact_graph_optimize(x, y, r, w, h, time_budget=15.0)

            # Step 3: SLSQP refinement
            x, y, r = slsqp_refine(x, y, r, w, h)

            # Step 4: Local search
            x, y, r = local_search(x, y, r, w, h, max_iters=100)

            # Step 5: Final SLSQP + expansion
            x, y, r = slsqp_refine(x, y, r, w, h)

            # Step 6: Expansion polish
            x, y, r = expansion_polish(x, y, r, w, h, time_budget=6.0)

            # Verify and record
            if is_valid(x, y, r, w, h):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_circles = np.column_stack([x, y, r])

    # Final polish on best result
    if best_circles is not None:
        x, y, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        # Determine w, h from the best result
        # We need to track which aspect ratio was best
        # For now, just try the polish with the stored values
        # Re-derive w, h from the bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        # The rectangle should have perimeter 4, so w + h = 2
        # We'll use the aspect ratio that was used
        # For simplicity, try a few aspect ratios
        for w_try in [1.0, 0.8, 1.25, 0.7, 1.4]:
            h_try = 2.0 - w_try
            if w_try < 0.4 or h_try < 0.4:
                continue
            # Scale positions to fit
            if x_max > w_try or y_max > h_try:
                continue
            r_polished = expand_radii(x, y, r, w_try, h_try, max_iters=200)
            if is_valid(x, y, r_polished, w_try, h_try):
                val = np.sum(r_polished)
                if val > best_value:
                    best_value = val
                    best_circles = np.column_stack([x, y, r_polished])

    if best_circles is None:
        # Fallback: simple grid
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        grid_x = np.array([((i % side) + 0.5) / side for i in range(N)])
        grid_y = np.array([((i // side) + 0.5) / side for i in range(N)])
        grid_r = np.full(N, 0.5 / side - 0.01)
        best_circles = np.column_stack([grid_x, grid_y, grid_r])

    return best_circles
