import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.4 or h < 0.4:
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty (squared)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                dist_req = r[i] + r[j]
                if d2 < dist_req * dist_req:
                    d = np.sqrt(d2)
                    obj += 500.0 * (dist_req - d) ** 2

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += 500.0 * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += 500.0 * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += 500.0 * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += 500.0 * (y[i] - (h - r[i])) ** 2

        # Keep aspect ratio reasonable
        if w < 0.5:
            obj += 500.0 * (0.5 - w) ** 2
        elif w > 1.5:
            obj += 500.0 * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-8):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        return True, float(np.sum(r))

    def get_initial_guesses():
        guesses = []
        rng = np.random.default_rng(42)

        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        side = int(np.ceil(np.sqrt(N)))

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        # Random starts
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    def force_relax(z, max_iter=150):
        z = z.copy().astype(float)
        n = N
        dt = 0.003

        for _ in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w

            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 20
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 20
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 20
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 20

            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 10
                        fy[i] += (dy / d) * overlap * 10
                        fx[j] -= (dx / d) * overlap * 10
                        fy[j] -= (dy / d) * overlap * 10

            # Update positions
            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            # Clamp positions
            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def polish(z, time_budget):
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        dt = 0.001
        start = time.time()

        for it in range(500):
            if time.time() - start > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 50
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 50
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 50
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 50

            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 20
                        fy[i] += (dy / d) * overlap * 20
                        fx[j] -= (dx / d) * overlap * 20
                        fy[j] -= (dy / d) * overlap * 20

            # Update positions
            x += dt * fx
            y += dt * fy

            # Try to grow radii slightly
            growth = 0.0001
            for i in range(n):
                can_grow = True
                if x[i] - (r[i] + growth) < 0 or x[i] + (r[i] + growth) > w:
                    can_grow = False
                if y[i] - (r[i] + growth) < 0 or y[i] + (r[i] + growth) > h:
                    can_grow = False
                if can_grow:
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r[i] + r[j] + 2 * growth:
                            can_grow = False
                            break
                if can_grow:
                    r[i] += growth

            # Clamp positions
            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        z_polished = np.zeros_like(z)
        z_polished[0::3][:n] = x
        z_polished[1::3][:n] = y
        z_polished[2::3][:n] = r
        z_polished[-1] = w

        return z_polished

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break

        z0_relaxed = force_relax(z0, max_iter=150)

        remaining = deadline - time.time()
        if remaining < 2:
            break

        maxiter = int(min(200, remaining * 25))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            polished = polish(best_result, remaining)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    r = r * (1 - 1e-6)

    return np.column_stack((x, y, r))
