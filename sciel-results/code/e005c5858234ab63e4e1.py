"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Polish."""
import numpy as np
import time

N = 21
_IU = np.triu_indices(N, k=1)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def objective(z):
        """Smooth penalty objective: minimize -sum(r) + penalties."""
        n = N
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return 1e6
        obj = -np.sum(r)
        # Pairwise overlap penalty
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(dx * dx + dy * dy)
        r_sum = r[_IU[0]] + r[_IU[1]]
        overlap = np.maximum(r_sum - d, 0.0)
        obj += 200.0 * np.sum(overlap * overlap)
        # Boundary penalty
        obj += 200.0 * np.sum(np.maximum(r - x, 0.0) ** 2)
        obj += 200.0 * np.sum(np.maximum(x + r - w, 0.0) ** 2)
        obj += 200.0 * np.sum(np.maximum(r - y, 0.0) ** 2)
        obj += 200.0 * np.sum(np.maximum(y + r - h, 0.0) ** 2)
        return obj

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[_IU[0]] + r[_IU[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def project_constraints(x, y, r, w, h, max_iter=500, dt=0.003):
        """Force-directed projection to resolve overlaps and boundary violations."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            fx += np.where(x < r, (r - x) * 80, 0.0)
            fx -= np.where(x > w - r, (x - (w - r)) * 80, 0.0)
            fy += np.where(y < r, (r - y) * 80, 0.0)
            fy -= np.where(y > h - r, (y - (h - r)) * 80, 0.0)

            # Pairwise repulsion
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[_IU[0]] + r[_IU[1]]
            overlap = np.maximum(r_sum - d, 0.0)

            if np.max(overlap) < 1e-10:
                break

            mask = overlap > 1e-12
            if np.any(mask):
                idx_i = _IU[0][mask]
                idx_j = _IU[1][mask]
                d_m = d[mask]
                ov_m = overlap[mask]
                ux = dx[mask] / np.maximum(d_m, 1e-10)
                uy = dy[mask] / np.maximum(d_m, 1e-10)
                f = ov_m * 30
                np.add.at(fx, idx_i, ux * f)
                np.add.at(fy, idx_i, uy * f)
                np.add.at(fx, idx_j, -ux * f)
                np.add.at(fy, idx_j, -uy * f)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y, r

    def expand_radii(x, y, r, w, h, max_rounds=60):
        """Iteratively expand each circle to its maximum possible radius."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for rnd in range(max_rounds):
            improved = False
            for i in range(n):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def polish(x, y, r, w, h, time_budget=10):
        """Alternating: expand radii, then project, then expand again."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        end_time = time.time() + time_budget
        for _ in range(200):
            if time.time() > end_time:
                break
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=30)
            x, y, r = project_constraints(x, y, r, w, h, max_iter=200, dt=0.002)
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=30)
        return x, y, r

    def simulated_annealing(z0, T0=0.5, T1=0.001, n_steps=20000, time_budget=60):
        """Simulated annealing on the full configuration vector."""
        n = N
        z = z0.copy()
        h = 2.0 - z[-1]
        cur_obj = objective(z)
        best_obj = cur_obj
        best_z = z.copy()
        end_time = time.time() + time_budget

        # Determine step sizes
        step_xy = 0.03
        step_r = 0.015
        step_w = 0.02

        for step in range(n_steps):
            if time.time() > end_time:
                break

            T = T0 * (T1 / T0) ** (step / max(n_steps - 1, 1))

            # Random perturbation
            z_new = z.copy()
            # Perturb 2-4 random circles' positions
            n_perturb = np.random.randint(2, 5)
            idx = np.random.choice(n, n_perturb, replace=False)
            z_new[idx * 1 + 0] += np.random.normal(0, step_xy, n_perturb)  # x
            z_new[idx * 1 + 1] += np.random.normal(0, step_xy, n_perturb)  # y
            # Perturb 1-2 random radii
            n_r_perturb = np.random.randint(1, 3)
            idx_r = np.random.choice(n, n_r_perturb, replace=False)
            z_new[idx_r * 1 + 2] += np.random.normal(0, step_r, n_r_perturb)
            # Sometimes perturb width
            if np.random.random() < 0.1:
                z_new[-1] += np.random.normal(0, step_w)

            # Clip to reasonable bounds
            z_new[-1] = np.clip(z_new[-1], 0.3, 1.7)
            for i in range(n):
                r_i = z_new[i * 3 + 2]
                if r_i < 0.01:
                    z_new[i * 3 + 2] = 0.01
                w_i = z_new[-1]
                h_i = 2.0 - w_i
                z_new[i * 3 + 0] = np.clip(z_new[i * 3 + 0], z_new[i * 3 + 2], w_i - z_new[i * 3 + 2])
                z_new[i * 3 + 1] = np.clip(z_new[i * 3 + 1], z_new[i * 3 + 2], h_i - z_new[i * 3 + 2])

            new_obj = objective(z_new)
            delta = new_obj - cur_obj

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                z = z_new
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_z = z.copy()

            # Anneal step size
            if step % 5000 == 4999 and step > 0:
                step_xy *= 0.8
                step_r *= 0.8
                step_w *= 0.8

        return best_z, best_obj

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        rng = np.random.default_rng(42)

        # Grid-based starts with various aspect ratios
        side = 5  # 5x5 grid, use 21 of 25 cells
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.65, 1.4, 0.6]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            z[-1] = w0
            # Place 21 circles in a 5x5 grid pattern (skip 4 corners for variety)
            positions = []
            for row in range(5):
                for col in range(5):
                    if row == 0 and col == 0:
                        continue
                    if row == 4 and col == 4:
                        continue
                    if row == 0 and col == 4:
                        continue
                    if row == 4 and col == 0:
                        continue
                    positions.append((row, col))
            positions = positions[:N]
            for i, (row, col) in enumerate(positions):
                cx = (col + 0.5) / 5.0 * w0
                cy = (row + 0.5) / 5.0 * h0
                r0 = min(w0 / 5.0, h0 / 5.0) * 0.45
                z[i * 3 + 0] = cx
                z[i * 3 + 1] = cy
                z[i * 3 + 2] = r0
            guesses.append(z)

        # Random starts
        for _ in range(8):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            z[-1] = w0
            for i in range(N):
                r0 = rng.uniform(0.03, 0.10)
                z[i * 3 + 0] = rng.uniform(r0, w0 - r0)
                z[i * 3 + 1] = rng.uniform(r0, h0 - r0)
                z[i * 3 + 2] = r0
            guesses.append(z)

        return guesses

    best_value = 0.0
    best_result = None

    guesses = get_initial_guesses()
    # Sort by closeness to square
    guesses.sort(key=lambda z: abs(z[-1] - 1.0))

    for gi, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Simulated annealing
        time_budget = min(40, deadline - (time.time() - start_time) - 15)
        if time_budget < 10:
            break
        z_best, obj_best = simulated_annealing(z0, T0=0.4, T1=0.0005,
                                               n_steps=15000, time_budget=time_budget)

        # Extract and project
        x = z_best[0::3][:N].copy()
        y = z_best[1::3][:N].copy()
        r = z_best[2::3][:N].copy()
        w = z_best[-1]
        h = 2.0 - w

        # Project to feasibility
        x, y, r = project_constraints(x, y, r, w, h, max_iter=300, dt=0.003)

        # Expand
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=50)

        # Polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            x, y, r = polish(x, y, r, w, h, time_budget=min(10, remaining - 5))

        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish on best
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        x, y, r = polish(x, y, r, w, h, time_budget=min(5, deadline - time.time() - 2))
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = 5
        cells = []
        count = 0
        for row in range(5):
            for col in range(5):
                if count >= N:
                    break
                cx = (col + 0.5) / side
                cy = (row + 0.5) / side
                r0 = 0.5 / side - 0.01
                cells.append((cx, cy, r0))
                count += 1
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
