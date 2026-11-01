"""Circle packing in perimeter-4 rectangle: Multi-phase optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
# Precompute upper triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def force_relax(x, y, r, w, h, max_iter=1000, dt=0.005):
        """Force-directed relaxation to resolve overlaps (vectorized)."""
        x = x.copy()
        y = y.copy()
        n = N
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion (vectorized)
            fx += np.where(x < r, (r - x) * 50, 0.0)
            fx -= np.where(x > w - r, (x - (w - r)) * 50, 0.0)
            fy += np.where(y < r, (r - y) * 50, 0.0)
            fy -= np.where(y > h - r, (y - (h - r)) * 50, 0.0)

            # Pairwise repulsion (vectorized)
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[_IU[0]] + r[_IU[1]]
            overlap = np.maximum(r_sum - d, 0.0)
            max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0

            if max_overlap < 1e-9:
                break

            # Compute forces for overlapping pairs
            mask = overlap > 1e-10
            if np.any(mask):
                idx_i = _IU[0][mask]
                idx_j = _IU[1][mask]
                d_masked = d[mask]
                ov_masked = overlap[mask]
                ux = dx[mask] / np.maximum(d_masked, 1e-10)
                uy = dy[mask] / np.maximum(d_masked, 1e-10)
                f = ov_masked * 20
                np.add.at(fx, idx_i, ux * f)
                np.add.at(fy, idx_i, uy * f)
                np.add.at(fx, idx_j, -ux * f)
                np.add.at(fy, idx_j, -uy * f)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y

    def expand_radii(x, y, r, w, h, max_rounds=50):
        """Iteratively try to expand each circle's radius (vectorized)."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for rnd in range(max_rounds):
            improved = False
            for i in range(n):
                # Max radius limited by walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                # Max radius limited by other circles (vectorized)
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf  # Exclude self
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def _lse(v, scale=50.0):
        """Smooth log-sum-exp approximation of max(v, 0)."""
        v = np.asarray(v, dtype=float)
        m = np.max(v)
        if m < -50.0 / scale:
            return 0.0
        return (1.0 / scale) * np.log(np.sum(np.exp(scale * (v - m)))) + m

    def slsqp_optimize(x0, y0, r0, w0, h0, time_budget=30):
        """SLSQP optimization with smooth log-sum-exp epigraph objective."""
        n = N
        z0 = np.zeros(3 * n + 1)
        z0[0::3][:n] = x0
        z0[1::3][:n] = y0
        z0[2::3][:n] = r0
        z0[-1] = w0

        def objective(z):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w
            if w < 0.1 or h < 0.1 or np.any(r < 0):
                return 1e6
            obj = -np.sum(r)
            # Smooth overlap penalty via log-sum-exp
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[_IU[0]] + r[_IU[1]]
            # violation = r_sum - d (positive when overlapping)
            overlap = r_sum - d
            obj += 10.0 * _lse(overlap, scale=80.0)
            # Smooth boundary penalties
            obj += 10.0 * _lse(r - x, scale=80.0)
            obj += 10.0 * _lse(x + r - w, scale=80.0)
            obj += 10.0 * _lse(r - y, scale=80.0)
            obj += 10.0 * _lse(y + r - h, scale=80.0)
            return obj

        bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
        maxiter = int(min(400, time_budget * 25))
        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            z_opt = res.x
            x = z_opt[0::3][:n]
            y = z_opt[1::3][:n]
            r = z_opt[2::3][:n]
            w = z_opt[-1]
            h = 2.0 - w
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
            # Try repair with slight shrink
            for shrink in [0.98, 0.95, 0.90]:
                r_try = r * shrink
                valid, val = check_validity(x, y, r_try, w, h)
                if valid:
                    return x, y, r_try, w, h, val
            return None
        except Exception:
            return None

    def coordinate_ascent_polish(x, y, r, w, h, time_budget=5):
        """Per-circle coordinate ascent: adjust position and radius for each circle."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        end_time = time.time() + time_budget
        for _ in range(100):
            if time.time() > end_time:
                break
            improved = False
            for i in range(n):
                # Try to increase radius of circle i
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                r_max = max(r_max, 0.0)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    improved = True

                # Try small position adjustments to free up space
                for dx_step in [0.005, -0.005, 0.002, -0.002]:
                    for dy_step in [0.005, -0.005, 0.002, -0.002]:
                        nx = x[i] + dx_step
                        ny = y[i] + dy_step
                        if nx < r[i] or nx > w - r[i]:
                            continue
                        if ny < r[i] or ny > h - r[i]:
                            continue
                        # Check separation
                        ddx = x - nx
                        ddy = y - ny
                        d2 = ddx * ddx + ddy * ddy
                        d2[i] = np.inf
                        r_sum = r + r[i]
                        r_sum[i] = 0
                        if np.all(d2 >= r_sum * r_sum - 1e-10):
                            x[i] = nx
                            y[i] = ny
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return x, y, r

    def local_search(x, y, r, w, h, time_budget=10):
        """Local search: try small moves to improve packing (faster validity check)."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        step_sizes = [0.02, 0.01, 0.005, 0.002]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]

        end_time = time.time() + time_budget
        for _ in range(200):
            if time.time() > end_time:
                break
            improved = False
            for i in range(n):
                for step in step_sizes:
                    for dx, dy in directions:
                        new_x = x[i] + dx * step
                        new_y = y[i] + dy * step
                        # Check if move is valid
                        if new_x < r[i] or new_x > w - r[i]:
                            continue
                        if new_y < r[i] or new_y > h - r[i]:
                            continue
                        # Vectorized validity check against all other circles
                        ddx = x - new_x
                        ddy = y - new_y
                        d2 = ddx * ddx + ddy * ddy
                        r_sum = r + r[i]
                        # Exclude self
                        d2[i] = np.inf
                        r_sum[i] = 0
                        if np.all(d2 >= r_sum * r_sum - 1e-10):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return x, y, r

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 1.3, 0.75]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        # Random starts with better spread
        rng = np.random.default_rng(123)
        for _ in range(10):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.035, 0.11)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by aspect ratio closeness to 1.0 (square tends to pack better)
    guesses.sort(key=lambda g: abs(g[3] - 1.0))

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        # Phase 1: Force relaxation
        x, y = force_relax(x0, y0, r0, w0, h0, max_iter=200)

        # Phase 2: SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        time_budget = min(20, remaining - 20)
        if time_budget > 5:
            result = slsqp_optimize(x, y, r0, w0, h0, time_budget)
            if result is not None:
                x, y, r, w, h, val = result
            else:
                x, y, r, w, h = x, y, r0, w0, h0
                valid, val = check_validity(x, y, r, w, h)
                if not valid:
                    continue
        else:
            valid, val = check_validity(x, y, r0, w0, h0)
            if not valid:
                continue
            r = r0.copy()
            w, h = w0, h0

        # Phase 3: Expand radii
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=40)

        # Phase 4: Coordinate ascent polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            x, y, r = coordinate_ascent_polish(x, y, r, w, h, time_budget=min(8, remaining - 15))
            # Expand again after coordinate ascent
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=30)

        # Phase 5: Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(5, remaining - 10))
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=25)

        # Check final validity
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 8:
        x, y, r, w, h = best_result
        # Coordinate ascent for fine-tuning
        remaining = deadline - (time.time() - start_time)
        if remaining > 6:
            x, y, r = coordinate_ascent_polish(x, y, r, w, h, time_budget=min(5, remaining - 6))
        # Try more expansion
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=60)
        # Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 4:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(3, remaining - 4))
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=40)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
