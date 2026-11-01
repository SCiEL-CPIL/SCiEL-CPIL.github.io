import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140

    pair_i, pair_j = np.triu_indices(N, k=1)

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = (xy[:, 0][:, None] - xy[:, 0][None, :])[pair_i, pair_j]
        dy = (xy[:, 1][:, None] - xy[:, 1][None, :])[pair_i, pair_j]
        d2 = dx*dx + dy*dy
        min_d = r[pair_i] + r[pair_j]
        if np.any(d2 < (min_d - tol)**2):
            return False
        return True

    def get_score(result):
        return np.sum(result[2*N:])

    def physics_optimize(params, max_iter=300, alpha=200.0, beta=100.0):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        pi, pj = pair_i, pair_j

        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)
            dx = xy[pj, 0] - xy[pi, 0]
            dy = xy[pj, 1] - xy[pi, 1]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-14))
            min_d = r[pi] + r[pj]
            margin = min_d - d
            # repulsion when overlapping or close
            force_mag = np.where(margin > 0, alpha * margin, alpha * 0.05 / (np.maximum(margin, 0) + 0.05)**2)
            # only apply when reasonably close to avoid global coupling
            close = d < min_d * 2.0
            force_mag = np.where(close, force_mag, 0.0)
            ux = dx / d
            uy = dy / d
            np.add.at(fx, pi, -force_mag * ux)
            np.add.at(fx, pj, force_mag * ux)
            np.add.at(fy, pi, -force_mag * uy)
            np.add.at(fy, pj, force_mag * uy)

            # boundary
            dl = xy[:, 0] - r
            dr = 1 - xy[:, 0] - r
            db = xy[:, 1] - r
            dt = 1 - xy[:, 1] - r
            fx += beta * np.maximum(0, 0.03 - dl)
            fx -= beta * np.maximum(0, 0.03 - dr)
            fy += beta * np.maximum(0, 0.03 - db)
            fy -= beta * np.maximum(0, 0.03 - dt)

            # gentle centering
            fx += 0.05 * (0.5 - xy[:, 0])
            fy += 0.05 * (0.5 - xy[:, 1])

            lr = 0.02 / (1 + it * 0.02)
            xy[:, 0] += lr * fx
            xy[:, 1] += lr * fy
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)

            if np.max(np.abs(fx)) < 1e-7 and np.max(np.abs(fy)) < 1e-7:
                break

        return np.concatenate([xy.flatten(), r])

    def push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        pi, pj = pair_i, pair_j

        for _ in range(n_iters):
            moved = False
            dx = xy[pj, 0] - xy[pi, 0]
            dy = xy[pj, 1] - xy[pi, 1]
            d = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-16))
            min_d = r[pi] + r[pj]
            overlap = min_d - d
            active = (overlap > 0) & (d > 1e-12)
            if np.any(active):
                wi = r[pi][active] / (r[pi][active] + r[pj][active])
                wj = 1 - wi
                ux = dx[active] / d[active]
                uy = dy[active] / d[active]
                ov = overlap[active]
                ii = pi[active]
                jj = pj[active]
                np.add.at(xy, ii, -np.stack([ux*ov*wi, uy*ov*wi], axis=1))
                np.add.at(xy, jj, np.stack([ux*ov*wj, uy*ov*wj], axis=1))
                moved = True
            # boundary clip
            lo = r
            hi = 1 - r
            if np.any(xy[:, 0] < lo) or np.any(xy[:, 0] > hi):
                xy[:, 0] = np.clip(xy[:, 0], lo, hi)
                moved = True
            if np.any(xy[:, 1] < lo) or np.any(xy[:, 1] > hi):
                xy[:, 1] = np.clip(xy[:, 1], lo, hi)
                moved = True
            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def smooth_penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        eps = 1e-8
        def smsq(a, b):
            diff = a - b
            sm = 0.5 * (diff + np.sqrt(diff*diff + eps))
            return np.sum(sm*sm)
        boundary_viol = smsq(r, xy[:, 0]) + smsq(r, 1 - xy[:, 0]) + smsq(r, xy[:, 1]) + smsq(r, 1 - xy[:, 1])
        dx = (xy[:, 0][:, None] - xy[:, 0][None, :])[pair_i, pair_j]
        dy = (xy[:, 1][:, None] - xy[:, 1][None, :])[pair_i, pair_j]
        dist = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-14))
        min_dist = r[pair_i] + r[pair_j]
        diff = min_dist - dist
        sm = 0.5 * (diff + np.sqrt(diff*diff + eps))
        overlap_viol = np.sum(sm*sm)
        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = (np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) +
                         np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2))
        dx = (xy[:, 0][:, None] - xy[:, 0][None, :])[pair_i, pair_j]
        dy = (xy[:, 1][:, None] - xy[:, 1][None, :])[pair_i, pair_j]
        dist = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-14))
        min_dist = r[pair_i] + r[pair_j]
        overlap = np.maximum(0, min_dist - dist)
        overlap_viol = np.sum(overlap*overlap)
        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def refine_solution(params, max_iter=400, tol=1e-13, use_smooth=True):
        try:
            obj = smooth_penalty_function if use_smooth else penalty_function
            result = minimize(obj, params, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def physics_refine(params, max_iter=250):
        phys_result = physics_optimize(params, max_iter=max_iter)
        if is_valid(phys_result):
            polished = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=True)
            if is_valid(polished):
                return polished
            return phys_result
        else:
            repaired = push_apart(phys_result, n_iters=100)
            if is_valid(repaired):
                polished = refine_solution(repaired, max_iter=250, tol=1e-13, use_smooth=True)
                if is_valid(polished):
                    return polished
            return params

    def hex_config(rng):
        n = N
        # hexagonal packing, 8 cols x 4 rows
        cols = 8
        rows = 4
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        spacing = 1.0 / cols
        h = spacing * np.sqrt(3) / 2
        total_h = h * (rows - 1)
        y0 = (1 - total_h) / 2
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing + (spacing/2 if row % 2 == 1 else 0)
            y = y0 + row * h
            xy[i] = [x, y]
            r[i] = spacing / 2 - 0.005
        xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        r += rng.uniform(-0.005, 0.005, size=n)
        r = np.clip(r, 0.02, 0.2)
        return np.concatenate([xy.flatten(), r])

    def grid_config(rng):
        n = N
        side = int(np.ceil(np.sqrt(n)))
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.5/side - 0.005
        xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        r += rng.uniform(-0.005, 0.005, size=n)
        r = np.clip(r, 0.02, 0.2)
        return np.concatenate([xy.flatten(), r])

    def random_config(rng, rmin, rmax):
        n = N
        xy = rng.uniform(0.08, 0.92, size=(n, 2))
        r = rng.uniform(rmin, rmax, size=n)
        return np.concatenate([xy.flatten(), r])

    def get_initial_config(restart, rng):
        m = restart % 6
        if m == 0:
            return hex_config(rng)
        elif m == 1:
            return grid_config(rng)
        elif m == 2:
            return random_config(rng, 0.05, 0.12)
        elif m == 3:
            return random_config(rng, 0.03, 0.08)
        elif m == 4:
            return hex_config(rng)
        else:
            return grid_config(rng)

    rng = np.random.RandomState(12345)

    best_result = None
    best_score = -np.inf

    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=80)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # primary refinement
        use_smooth = (restart % 2 == 0)
        refined = refine_solution(params, max_iter=350, tol=1e-13, use_smooth=use_smooth)

        # physics refinement on a subset
        if restart % 3 == 0 and time.time() < deadline - 25:
            phys_refined = physics_refine(params, max_iter=200)
            if is_valid(phys_refined):
                ps = get_score(phys_refined)
                if ps > best_score:
                    best_score = ps
                    best_result = phys_refined.copy()

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()

    # Final polish phase
    if best_result is not None and time.time() < deadline - 15:
        polished = refine_solution(best_result, max_iter=1000, tol=1e-15, use_smooth=False)
        if is_valid(polished) and get_score(polished) > best_score:
            best_score = get_score(polished)
            best_result = polished.copy()

        polished_smooth = refine_solution(best_result, max_iter=1000, tol=1e-15, use_smooth=True)
        if is_valid(polished_smooth) and get_score(polished_smooth) > best_score:
            best_score = get_score(polished_smooth)
            best_result = polished_smooth.copy()

        if time.time() < deadline - 20:
            phys_polished = physics_refine(best_result, max_iter=300)
            if is_valid(phys_polished) and get_score(phys_polished) > best_score:
                best_score = get_score(phys_polished)
                best_result = phys_polished.copy()

        # Perturbation-based polish
        for polish_iter in range(20):
            if time.time() > deadline - 5:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.004, 0.004, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=50)
            if polish_iter % 2 == 0:
                polished2 = refine_solution(perturbed, max_iter=500, tol=1e-15, use_smooth=False)
            else:
                polished2 = physics_refine(perturbed, max_iter=200)
            if is_valid(polished2) and get_score(polished2) > best_score:
                best_score = get_score(polished2)
                best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
