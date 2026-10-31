"""Circle packing in perimeter-4 rectangle: Multi-phase optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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
        """Force-directed relaxation to resolve overlaps."""
        x = x.copy()
        y = y.copy()
        n = N
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 50
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 50
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 50
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 50

            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        overlap = r_sum - d
                        max_overlap = max(max_overlap, overlap)
                        if d > 1e-10:
                            ux = dx / d
                            uy = dy / d
                            f = overlap * 20
                            fx[i] += ux * f
                            fy[i] += uy * f
                            fx[j] -= ux * f
                            fy[j] -= uy * f

            if max_overlap < 1e-9:
                break

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y

    def expand_radii(x, y, r, w, h, max_rounds=50):
        """Iteratively try to expand each circle's radius."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for rnd in range(max_rounds):
            improved = False
            for i in range(n):
                # Max radius limited by walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                # Max radius limited by other circles
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    r_max = min(r_max, r_allowed)
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-8:
                    new_r = r_max
                    # Check validity
                    valid = True
                    if new_r > x[i] or new_r > w - x[i] or new_r > y[i] or new_r > h - y[i]:
                        valid = False
                    else:
                        for j in range(n):
                            if i == j:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d = np.sqrt(dx * dx + dy * dy)
                            if d < new_r + r[j] - 1e-10:
                                valid = False
                                break
                    if valid:
                        r[i] = new_r
                        improved = True
            if not improved:
                break
        return x, y, r

    def slsqp_optimize(x0, y0, r0, w0, h0, time_budget=30):
        """SLSQP optimization with penalty method."""
        n = N
        z0 = np.zeros(3 * n + 1)
        z0[0::3][:n] = x0
        z0[1::3][:n] = y0
        z0[2::3][:n] = r0
        z0[-1] = w0

        penalty_weight = 500.0

        def objective(z):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w
            if w < 0.1 or h < 0.1 or np.any(r < 0):
                return 1e6
            obj = -np.sum(r)
            # Overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = max(0, r[i] + r[j] - d)
                    obj += penalty_weight * overlap * overlap
            # Boundary penalty
            for i in range(n):
                obj += penalty_weight * max(0, r[i] - x[i]) ** 2
                obj += penalty_weight * max(0, x[i] + r[i] - w) ** 2
                obj += penalty_weight * max(0, r[i] - y[i]) ** 2
                obj += penalty_weight * max(0, y[i] + r[i] - h) ** 2
            return obj

        bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]
        maxiter = int(min(500, time_budget * 30))
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
            # Try repair
            r *= 0.95
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
            return None
        except Exception:
            return None

    def local_search(x, y, r, w, h, time_budget=10):
        """Local search: try small moves to improve packing."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        step_sizes = [0.02, 0.01, 0.005, 0.002]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]

        for _ in range(100):
            if time.time() > time_budget:
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
                        valid = True
                        for j in range(n):
                            if i == j:
                                continue
                            ddx = new_x - x[j]
                            ddy = new_y - y[j]
                            if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-10:
                                valid = False
                                break
                        if valid:
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

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Phase 1: Force relaxation
        x, y = force_relax(x0, y0, r0, w0, h0, max_iter=300)

        # Phase 2: SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        time_budget = min(25, remaining - 25)
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
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=30)

        # Phase 4: Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(8, remaining - 15))
            # Expand again after local search
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=20)

        # Check final validity
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w, h = best_result
        # Try more expansion
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=50)
        # Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(5, remaining - 5))
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=30)
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
