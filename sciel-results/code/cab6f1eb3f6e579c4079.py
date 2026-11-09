"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21
DEADLINE = 130  # seconds, safety margin before 150s limit


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

    def _force_relaxation(w_init, x0, y0, r0, max_iters=2000):
        """
        Physics-inspired force relaxation with hard constraint projection.
        Circles are pushed apart and expanded to maximize sum of radii.
        """
        h = 2.0 - w_init
        if h <= 0.01 or w_init <= 0.01:
            return None, -1.0

        xs = x0.copy().astype(float)
        ys = y0.copy().astype(float)
        rs = r0.copy().astype(float)
        w = w_init

        def _project_constraints():
            """Project circles to satisfy boundary and non-overlap constraints."""
            # Boundary projection
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    xs[i] = rs[i]
                if xs[i] + rs[i] > w:
                    xs[i] = w - rs[i]
                if ys[i] - rs[i] < 0:
                    ys[i] = rs[i]
                if ys[i] + rs[i] > h:
                    ys[i] = h - rs[i]

            # Non-overlap projection (iterative)
            for _ in range(50):
                moved = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[j] - xs[i]
                        dy = ys[j] - ys[i]
                        dist = math.sqrt(dx * dx + dy * dy)
                        min_d = rs[i] + rs[j]
                        if dist < min_d - 1e-12:
                            if dist < 1e-12:
                                dx = 0.01
                                dy = 0.0
                                dist = 0.01
                            overlap = (min_d - dist) / 2.0
                            nx = dx / dist
                            ny = dy / dist
                            # Move both circles apart
                            xs[i] -= overlap * nx
                            ys[i] -= overlap * ny
                            xs[j] += overlap * nx
                            ys[j] += overlap * ny
                            moved = True

                # Re-apply boundary constraints after moving
                for i in range(N):
                    if xs[i] - rs[i] < 0:
                        xs[i] = rs[i]
                    if xs[i] + rs[i] > w:
                        xs[i] = w - rs[i]
                    if ys[i] - rs[i] < 0:
                        ys[i] = rs[i]
                    if ys[i] + rs[i] > h:
                        ys[i] = h - rs[i]

                if not moved:
                    break

        def _grow_radii(delta):
            """Attempt to grow all radii by delta, maintaining validity."""
            nonlocal xs, ys, rs
            new_rs = rs + delta
            # Check if growing would cause violations
            valid = True
            for i in range(N):
                if xs[i] - new_rs[i] < -1e-10 or xs[i] + new_rs[i] > w + 1e-10:
                    valid = False
                    break
                if ys[i] - new_rs[i] < -1e-10 or ys[i] + new_rs[i] > h + 1e-10:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                        if dist < new_rs[i] + new_rs[j] - 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
            return valid

        best_local_score = -1.0
        best_local_xs = xs.copy()
        best_local_ys = ys.copy()
        best_local_rs = rs.copy()

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break

            # Compute current score
            current_score = np.sum(rs)
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_xs = xs.copy()
                best_local_ys = ys.copy()
                best_local_rs = rs.copy()

            # Try to grow radii
            delta = 0.002
            if _grow_radii(delta):
                rs += delta
                continue

            # Force relaxation: push apart overlapping circles
            # This allows the system to find new configurations where radii can grow
            _project_constraints()
            
            # Add small random perturbations to escape local minima
            if it % 10 == 0:
                for i in range(N):
                    xs[i] += np.random.uniform(-0.01, 0.01)
                    ys[i] += np.random.uniform(-0.01, 0.01)
                _project_constraints()

        # Final: try to grow radii as much as possible
        for _ in range(1000):
            if time.time() - start_time > DEADLINE - 10:
                break
            delta = 0.001
            if _grow_radii(delta):
                rs += delta
            else:
                delta = 0.0001
                if _grow_radii(delta):
                    rs += delta
                else:
                    break

        circles = np.column_stack([xs, ys, rs])
        _update_best(circles, w, h)
        return circles, _validate_and_score(circles)

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
        if time.time() - start_time > DEADLINE - 30:
            break

        h = 2.0 / (ar + 1.0)
        w = ar * h

        positions, radii = _grid_packing(rows, cols, w, h)
        if positions is not None:
            circles = np.column_stack([positions, radii])
            _update_best(circles, w, h)

    # Phase 2: Force relaxation from best grid
    if best_result is not None and time.time() - start_time < DEADLINE - 40:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]

        w_est = max(np.max(xs + rs), 0.5)
        h_est = max(np.max(ys + rs), 0.5)
        scale = 2.0 / (w_est + h_est)
        w_sa = w_est * scale
        h_sa = h_est * scale

        if 0.3 < w_sa < 1.7:
            _force_relaxation(w_sa, xs, ys, rs, max_iters=3000)

    # Phase 3: Force relaxation from multiple grid starts
    if time.time() - start_time < DEADLINE - 30:
        np.random.seed(42)
        for w_frac in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            if time.time() - start_time > DEADLINE - 30:
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

            _force_relaxation(w, x0, y0, r0, max_iters=2000)

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

            _force_relaxation(w, x0, y0, r0, max_iters=1500)

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
            n_var = 3 * N + 1

            def objective_polish(v):
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

            bounds_polish = []
            for i in range(N):
                bounds_polish.append((0, 2.0))
                bounds_polish.append((0, 2.0))
                bounds_polish.append((0, 1.0))
            bounds_polish.append((0.3, 1.7))

            v0_polish = np.zeros(n_var)
            for i in range(N):
                v0_polish[i*3] = xs[i]
                v0_polish[i*3+1] = ys[i]
                v0_polish[i*3+2] = rs[i]
            v0_polish[3*N] = w_polish

            try:
                res = minimize(objective_polish, v0_polish, method='SLSQP',
                              bounds=bounds_polish, options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < objective_polish(v0_polish):
                    xs_p = res.x[0:3*N:3]
                    ys_p = res.x[1:3*N:3]
                    rs_p = res.x[2:3*N:3]
                    w_p = res.x[3*N]
                    h_p = 2.0 - w_p
                    circles_p = np.column_stack([xs_p, ys_p, rs_p])
                    _update_best(circles_p, w_p, h_p)
            except Exception:
                pass

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
