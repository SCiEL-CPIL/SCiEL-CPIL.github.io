"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21
DEADLINE = 140  # seconds, safety margin before 150s limit


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


def _physics_simulate(w, h, x0, y0, r0, n_steps=500, dt=0.01, cooling=0.999):
    """
    Physics-based soft-body simulation using repulsive potentials.
    Circles are treated as particles with repulsive forces between them
    and attractive forces to the walls.
    """
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)

    # Repulsion strength and wall attraction
    repulsion_k = 100.0
    wall_k = 50.0
    damping = 0.9

    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)

    for step in range(n_steps):
        if time.time() > DEADLINE:
            break

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]

                if dist < min_dist and dist > 1e-10:
                    # Strong repulsion when overlapping
                    overlap = min_dist - dist
                    force = repulsion_k * overlap * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

                    # Slightly reduce radii to allow packing
                    fr[i] -= 0.01 * overlap
                    fr[j] -= 0.01 * overlap
                elif dist > 0.1 and dist < 1.0:
                    # Mild repulsion at close range
                    force = repulsion_k * 0.01 * (1.0 - dist)
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Wall attraction (keep circles inside)
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                fx[i] += wall_k * (r[i] - x[i])
            # Right wall
            if x[i] > w - r[i]:
                fx[i] -= wall_k * (x[i] - (w - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += wall_k * (r[i] - y[i])
            # Top wall
            if y[i] > h - r[i]:
                fy[i] -= wall_k * (y[i] - (h - r[i]))

            # Radius growth tendency (try to grow)
            fr[i] += 0.1

            # Keep radius positive
            if r[i] < 0.01:
                fr[i] += 10.0

        # Update velocities with damping
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        vr = vr * damping + fr * dt

        # Update positions
        x = x + vx * dt
        y = y + vy * dt
        r = r + vr * dt

        # Clamp positions to stay inside
        for i in range(N):
            x[i] = max(r[i], min(x[i], w - r[i]))
            y[i] = max(r[i], min(y[i], h - r[i]))
            r[i] = max(0.01, min(r[i], min(w, h) / 2.0))

        # Cool down
        dt *= cooling

    return x, y, r


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

    def _run_optimization(w_init, x0, y0, r0, max_iter=200, ftol=1e-10):
        """Run SLSQP optimization for given initial conditions."""
        h = 2.0 - w_init
        if h <= 0.01 or w_init <= 0.01:
            return None, -1.0

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.01 or w <= 0.01:
                return 1e10
            # Penalty for out-of-bounds
            penalty = 0.0
            for i in range(N):
                if xs[i] - rs[i] < -1e-6:
                    penalty += 100 * (xs[i] - rs[i] + 1e-6)**2
                if xs[i] + rs[i] > w + 1e-6:
                    penalty += 100 * (xs[i] + rs[i] - w - 1e-6)**2
                if ys[i] - rs[i] < -1e-6:
                    penalty += 100 * (ys[i] - rs[i] + 1e-6)**2
                if ys[i] + rs[i] > h + 1e-6:
                    penalty += 100 * (ys[i] + rs[i] - h - 1e-6)**2
            # Non-overlap penalty
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 100 * (min_d - dist)**2
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((0, 1.0))  # r
        bounds.append((0.3, 1.7))  # w

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': ftol})
            if res.success or res.fun < objective(v0):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                h = 2.0 - w
                circles = np.column_stack([xs, ys, rs])
                _update_best(circles, w, h)
                return circles, _validate_and_score(circles)
        except Exception:
            pass

        return None, -1.0

    # Phase 1: Grid-based packings with various aspect ratios
    candidate_configs = []
    for rows in range(1, 10):
        for cols in range(1, 10):
            if rows * cols >= N:
                ar = cols / rows
                candidate_configs.append((rows, cols, ar))

    # Add non-integer aspect ratios
    for ar in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0]:
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

    # Remove duplicates
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

    # Phase 2: Physics-based simulation from grid initializations
    if time.time() - start_time < DEADLINE - 20:
        for rows, cols, ar in unique_configs[:10]:  # Limit to top 10 configs
            if time.time() - start_time > DEADLINE - 20:
                break

            h = 2.0 / (ar + 1.0)
            w = ar * h

            positions, radii = _grid_packing(rows, cols, w, h)
            if positions is not None:
                # Run physics simulation
                x_sim, y_sim, r_sim = _physics_simulate(
                    w, h, positions[:, 0], positions[:, 1], radii,
                    n_steps=300, dt=0.005, cooling=0.9995
                )
                circles = np.column_stack([x_sim, y_sim, r_sim])
                _update_best(circles, w, h)

    # Phase 3: Optimization from grid initializations
    if time.time() - start_time < DEADLINE - 15:
        # Use the best grid result as starting point for optimization
        if best_result is not None:
            xs = best_result[:, 0]
            ys = best_result[:, 1]
            rs = best_result[:, 2]

            # Estimate w from bounding box
            w_est = max(np.max(xs + rs), 0.5)
            h_est = max(np.max(ys + rs), 0.5)
            scale = 2.0 / (w_est + h_est)
            w_polish = w_est * scale
            h_polish = h_est * scale

            if 0.3 < w_polish < 1.7:
                _run_optimization(w_polish, xs, ys, rs, max_iter=300, ftol=1e-10)

    # Phase 4: Try multiple grid-based starts with optimization
    if time.time() - start_time < DEADLINE - 10:
        np.random.seed(42)
        for w_frac in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            if time.time() - start_time > DEADLINE - 10:
                break

            w = w_frac
            h = 2.0 - w
            if h <= 0.1:
                continue

            # Arrange circles in a grid pattern
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
                r0[i] = min(w / (2 * cols), h / (2 * rows)) * 0.95

            _run_optimization(w, x0, y0, r0, max_iter=150, ftol=1e-9)

    # Phase 5: Random restarts if time permits
    if time.time() - start_time < DEADLINE - 5:
        np.random.seed(123)
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break

            w = np.random.uniform(0.7, 1.3)
            h = 2.0 - w
            if h <= 0.1:
                continue
            x0 = np.random.uniform(0.1, w - 0.1, N)
            y0 = np.random.uniform(0.1, h - 0.1, N)
            r0 = np.full(N, 0.08)

            _run_optimization(w, x0, y0, r0, max_iter=100, ftol=1e-8)

    # Phase 6: Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 2:
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
                # Non-overlap penalty
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

    # Fallback: return a valid grid configuration if nothing better found
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
