import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    best_result = None
    best_score = -np.inf

    def penalty_function(params, mu=50.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Smooth soft-min for overlap: use log-sum-exp style
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist2 = dx**2 + dy**2
        dist = np.sqrt(dist2 + 1e-15)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)

        # Soft overlap: -log(sum(exp(-mu*(dist-min_dist)))) approximates min
        # Use a quadratic soft version: sum(max(0, min_dist-dist)^2) but smoothed
        overlap_vec = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap_vec**2)

        # Boundary
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = (np.sum(np.maximum(0, -d_left)**2) +
                         np.sum(np.maximum(0, -d_right)**2) +
                         np.sum(np.maximum(0, -d_bottom)**2) +
                         np.sum(np.maximum(0, -d_top)**2))

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

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

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False

        return True

    def get_score(result):
        return np.sum(result[2*N:])

    def push_apart(params, n_iters=80):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist > 1e-12:
                            overlap = min_dist - dist
                            wi = r[i] / (r[i] + r[j])
                            wj = r[j] / (r[i] + r[j])
                            push_x = (dx / dist) * overlap
                            push_y = (dy / dist) * overlap
                            xy[i, 0] -= push_x * wi
                            xy[i, 1] -= push_y * wi
                            xy[j, 0] += push_x * wj
                            xy[j, 1] += push_y * wj
                        else:
                            angle = np.random.uniform(0, 2*np.pi)
                            xy[i, 0] -= 0.01 * np.cos(angle)
                            xy[i, 1] -= 0.01 * np.sin(angle)
                            xy[j, 0] += 0.01 * np.cos(angle)
                            xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def refine_solution(params, max_iter=300, tol=1e-12):
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
            return result.x
        except Exception:
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart == 0:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart == 1:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        elif restart == 2:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart == 3:
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart == 4:
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)
        elif restart == 5:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        elif restart == 6:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        elif restart == 7:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart == 8:
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        elif restart == 9:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        else:
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.04, 0.09, size=n)

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)

    # Phase 1: Multiple restarts
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=40)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        refined = refine_solution(params, max_iter=200)

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            repaired = push_apart(refined, n_iters=60)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                xy = refined[:2*N].reshape(N, 2)
                r = refined[2*N:]
                for _ in range(100):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                if is_valid(np.concatenate([xy.flatten(), r])):
                    score = get_score(np.concatenate([xy.flatten(), r]))
                    if score > best_score:
                        best_score = score
                        best_result = np.concatenate([xy.flatten(), r]).copy()

    # Phase 2: Polish
    if best_result is not None and time.time() < deadline - 10:
        polished = refine_solution(best_result, max_iter=500, tol=1e-14)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        for polish_iter in range(5):
            if time.time() > deadline:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2)
            r_p = perturbed[2*N:]
            xy_p += rng.uniform(-0.005, 0.005, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=30)
            polished2 = refine_solution(perturbed, max_iter=300, tol=1e-14)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2)
        r = best_result[2*n:]
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
