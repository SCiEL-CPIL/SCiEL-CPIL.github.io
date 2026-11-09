"""Circle packing in perimeter-4 rectangle: Combined Grid, Force-Relaxation, and SLSQP."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before 150s limit


def _validate_and_score(circles):
    """Validate circles and compute score (sum of radii)."""
    if len(circles) != N:
        return -1.0
    if np.any(circles[:, 2] < 0):
        return -1.0
    if np.any(np.isnan(circles)) or np.any(np.isinf(circles)):
        return -1.0
    return float(np.sum(circles[:, 2]))


def _check_validity(circles, w, h):
    """Check if circles are valid (non-overlapping, within bounds)."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < -1e-9 or x + r > w + 1e-9:
            return False
        if y - r < -1e-9 or y + r > h + 1e-9:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True


def construct():
    start_time = time.time()
    best_score = -1.0
    best_result = None

    def _update_best(circles, w, h):
        """Validate and update best result if valid."""
        nonlocal best_score, best_result
        if _check_validity(circles, w, h):
            score = _validate_and_score(circles)
            if score > best_score:
                best_score = score
                best_result = circles.copy()

    def _grid_packing(rows, cols, w, h):
        """
        Compute a grid-based packing for rows x cols grid in rectangle w x h.
        Returns (positions, radii) or (None, None) if invalid.
        """
        if rows * cols < N:
            return None, None

        positions = np.zeros((N, 2))

        for i in range(N):
            row = i // cols
            col = i % cols

            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols

            y = (row + 0.5) * h / rows

            positions[i] = [x, y]

        # Compute maximum valid radius
        min_r = float('inf')

        # Boundary constraints
        for i in range(N):
            min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                       positions[i, 1], h - positions[i, 1])

        # Pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_r = min(min_r, dist / 2.0)

        if min_r <= 0:
            return None, None

        radii = np.full(N, min_r)
        return positions, radii

    def _force_relax(positions, radii, w, h, max_steps=2000, time_budget=30.0):
        """
        Force-directed relaxation to maximize sum of radii.
        Combines radius growth with overlap/boundary resolution.
        """
        pos = positions.copy().astype(float)
        rad = radii.copy().astype(float)

        # Ensure initial validity
        for _ in range(20):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.9
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.9
                        rad[j] *= 0.9
                        valid = False
            if valid:
                break

        step = 0
        while step < max_steps:
            if time.time() - start_time > DEADLINE - 15:
                break

            # 1. Attempt to grow radii uniformly
            delta_r = 0.002
            rad_new = rad + delta_r
            valid = True
            for i in range(N):
                if pos[i, 0] - rad_new[i] < 0 or pos[i, 0] + rad_new[i] > w or pos[i, 1] - rad_new[i] < 0 or pos[i, 1] + rad_new[i] > h:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < rad_new[i] + rad_new[j]:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                rad = rad_new
                step += 1
                continue

            # 2. Resolve overlaps by pushing apart
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_d = rad[i] + rad[j]
                    if dist < min_d - 1e-12:
                        if dist < 1e-12:
                            dx, dy = 0.01, 0.0
                            dist = 0.01
                        overlap = (min_d - dist) / 2.0
                        nx, ny = dx / dist, dy / dist
                        pos[i, 0] -= overlap * nx
                        pos[i, 1] -= overlap * ny
                        pos[j, 0] += overlap * nx
                        pos[j, 1] += overlap * ny
                        moved = True

            # 3. Resolve boundary violations
            for i in range(N):
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]
                # If still invalid, shrink
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95

            # 4. If no movement and can't grow, try small random perturbation
            if not moved:
                i = np.random.randint(N)
                angle = np.random.uniform(0, 2*np.pi)
                step_size = 0.02
                pos[i, 0] += step_size * math.cos(angle)
                pos[i, 1] += step_size * math.sin(angle)
                # Re-clamp
                pos[i, 0] = max(rad[i], min(w - rad[i], pos[i, 0]))
                pos[i, 1] = max(rad[i], min(h - rad[i], pos[i, 1]))

            step += 1

        # Final cleanup
        for _ in range(10):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        return pos, rad

    def _slsqp_polish(xs, ys, rs, w_init, max_iter=300):
        """Polish with SLSQP."""
        h_init = 2.0 - w_init
        if h_init <= 0.1 or w_init <= 0.1:
            return None

        n_var = 3 * N + 1

        def objective(v):
            xs_p = v[0:3*N:3]
            ys_p = v[1:3*N:3]
            rs_p = v[2:3*N:3]
            w_p = v[3*N]
            h_p = 2.0 - w_p
            if h_p <= 0.01 or w_p <= 0.01:
                return 1e10
            penalty = 0.0
            for i in range(N):
                if xs_p[i] - rs_p[i] < -1e-8:
                    penalty += 1000 * (xs_p[i] - rs_p[i] + 1e-8)**2
                if xs_p[i] + rs_p[i] > w_p + 1e-8:
                    penalty += 1000 * (xs_p[i] + rs_p[i] - w_p - 1e-8)**2
                if ys_p[i] - rs_p[i] < -1e-8:
                    penalty += 1000 * (ys_p[i] - rs_p[i] + 1e-8)**2
                if ys_p[i] + rs_p[i] > h_p + 1e-8:
                    penalty += 1000 * (ys_p[i] + rs_p[i] - h_p - 1e-8)**2
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs_p[i] - xs_p[j])**2 + (ys_p[i] - ys_p[j])**2)
                    min_d = rs_p[i] + rs_p[j]
                    if dist < min_d:
                        penalty += 1000 * (min_d - dist)**2
            return -np.sum(rs_p) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((0, 1.0))
        bounds.append((0.3, 1.7))

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = xs[i]
            v0[i*3+1] = ys[i]
            v0[i*3+2] = rs[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12})
            if res.fun < objective(v0):
                xs_p = res.x[0:3*N:3]
                ys_p = res.x[1:3*N:3]
                rs_p = res.x[2:3*N:3]
                w_p = res.x[3*N]
                h_p = 2.0 - w_p
                circles_p = np.column_stack([xs_p, ys_p, rs_p])
                return circles_p, w_p, h_p
        except Exception:
            pass
        return None

    # Phase 1: Grid-based packings with various aspect ratios
    candidate_configs = []
    for rows in range(1, 10):
        for cols in range(1, 10):
            if rows * cols >= N:
                ar = cols / rows
                candidate_configs.append((rows, cols, ar))

    for ar in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.33, 2.5, 3.0]:
        best_rows, best_cols = None, None
        best_diff = float('inf')
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    diff = abs(grid_ar - ar)
                    if diff < best_diff:
                        best_diff = diff
                        best_rows, best_cols = rows, cols
        if best_rows is not None:
            candidate_configs.append((best_rows, best_cols, ar))

    seen = set()
    unique_configs = []
    for rows, cols, ar in candidate_configs:
        key = (rows, cols, round(ar, 3))
        if key not in seen:
            seen.add(key)
            unique_configs.append((rows, cols, ar))

    for rows, cols, ar in unique_configs:
        if time.time() - start_time > DEADLINE - 40:
            break

        h = 2.0 / (ar + 1.0)
        w = ar * h

        positions, radii = _grid_packing(rows, cols, w, h)
        if positions is not None:
            circles = np.column_stack([positions, radii])
            _update_best(circles, w, h)

    # Phase 2: Force relaxation from best grid
    if best_result is not None and time.time() - start_time < DEADLINE - 30:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]

        w_est = max(np.max(xs + rs), 0.5)
        h_est = max(np.max(ys + rs), 0.5)
        scale = 2.0 / (w_est + h_est)
        w_fr = w_est * scale
        h_fr = h_est * scale

        if 0.3 < w_fr < 1.7:
            pos_fr, rad_fr = _force_relax(xs, ys, rs, w_fr, h_fr, max_steps=3000)
            circles_fr = np.column_stack([pos_fr, rad_fr])
            _update_best(circles_fr, w_fr, h_fr)

    # Phase 3: Force relaxation from multiple grid starts
    if time.time() - start_time < DEADLINE - 25:
        np.random.seed(42)
        for w_frac in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            if time.time() - start_time > DEADLINE - 25:
                break

            w = w_frac
            h = 2.0 - w
            if h <= 0.1:
                continue

            cols = max(1, int(np.round(N * w / (w + h))))
            rows = max(1, int(np.ceil(N / cols)))

            x0 = np.zeros(N)
            y0 = np.zeros(N)
            r0 = np.zeros(N)

            for i in range(N):
                col = i % cols
                row = i // cols
                x0[i] = (col + 0.5) * w / cols
                y0[i] = (row + 0.5) * h / rows
                r0[i] = min(w / (2 * cols), h / (2 * rows)) * 0.9

            pos_fr, rad_fr = _force_relax(x0, y0, r0, w, h, max_steps=2000)
            circles_fr = np.column_stack([pos_fr, rad_fr])
            _update_best(circles_fr, w, h)

    # Phase 4: Random restarts with force relaxation
    if time.time() - start_time < DEADLINE - 20:
        np.random.seed(123)
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 20:
                break

            w = np.random.uniform(0.7, 1.3)
            h = 2.0 - w
            if h <= 0.1:
                continue
            x0 = np.random.uniform(0.05, w - 0.05, N)
            y0 = np.random.uniform(0.05, h - 0.05, N)
            r0 = np.full(N, 0.06)

            pos_fr, rad_fr = _force_relax(x0, y0, r0, w, h, max_steps=2000)
            circles_fr = np.column_stack([pos_fr, rad_fr])
            _update_best(circles_fr, w, h)

    # Phase 5: Final polish with SLSQP on best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]

        w_est = max(np.max(xs + rs), 0.5)
        h_est = max(np.max(ys + rs), 0.5)
        scale = 2.0 / (w_est + h_est)
        w_polish = w_est * scale
        h_polish = h_est * scale

        if 0.3 < w_polish < 1.7:
            result = _slsqp_polish(xs, ys, rs, w_polish, max_iter=200)
            if result is not None:
                circles_p, w_p, h_p = result
                _update_best(circles_p, w_p, h_p)

    # Fallback
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        best_result = np.array(cells)
        best_score = _validate_and_score(best_result)

    return best_result
