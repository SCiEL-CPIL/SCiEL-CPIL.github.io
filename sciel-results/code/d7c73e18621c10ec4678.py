import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_score = -np.inf
    best_result = None

    def is_valid(x, tol=1e-5):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]

        if W <= 0 or H <= 0:
            return False
        if 2*(W+H) > 4 + tol:
            return False

        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > W - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > H - rs[i] + tol:
                return False
            if rs[i] < 1e-8:
                return False

        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - tol:
                    return False
        return True

    def get_score(x):
        if not is_valid(x):
            return -np.inf
        return np.sum(x[2*N:3*N])

    def relax_radii(x, n_iter=20, tol=1e-9):
        """Iteratively increase each radius to its maximum given fixed positions."""
        x = x.copy()
        xs = x[:N].copy()
        ys = x[N:2*N].copy()
        rs = x[2*N:3*N].copy()
        W = x[3*N]
        H = x[3*N+1]

        for _ in range(n_iter):
            improved = False
            for i in range(N):
                r_bound = min(xs[i], W - xs[i], ys[i], H - ys[i])
                r_max = r_bound
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    r_candidate = d - rs[j]
                    if r_candidate < r_max:
                        r_max = r_candidate
                r_max = max(r_max, 1e-8)
                if r_max > rs[i] + tol:
                    rs[i] = r_max
                    improved = True
            if not improved:
                break
        x[2*N:3*N] = rs
        return x

    def objective(x):
        return -np.sum(x[2*N:3*N])

    def constraints_def(x):
        cons = []
        cons.append({'type': 'ineq', 'fun': lambda x: 2 - (x[3*N] + x[3*N+1])})
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N] - x[i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N+1] - x[N+i] - x[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i] - 1e-7})
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j:
                        (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
                })
        return cons

    def grid_init(W, H, cols=None, rows=None, noise=0.0):
        if cols is None:
            cols = int(np.ceil(np.sqrt(N)))
        if rows is None:
            rows = int(np.ceil(N / cols))
        dx = W / cols
        dy = H / rows
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N:
                    break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                rs[k] = 0.4 * min(dx, dy)
                k += 1
        if noise > 0 and k == N:
            xs += np.random.normal(0, noise, N)
            ys += np.random.normal(0, noise, N)
        return np.concatenate([xs, ys, rs, [W, H]])

    def random_init(W, H):
        xs = np.random.uniform(0.1, max(W-0.1, 0.11), N)
        ys = np.random.uniform(0.1, max(H-0.1, 0.11), N)
        rs = np.random.uniform(0.04, 0.15, N)
        return np.concatenate([xs, ys, rs, [W, H]])

    def force_relax(x, n_iter=50, cooling=True):
        """
        Force-directed relaxation: push overlapping circles apart,
        pull them toward center, and grow radii.
        """
        x = x.copy()
        xs = x[:N].copy()
        ys = x[N:2*N].copy()
        rs = x[2*N:3*N].copy()
        W = x[3*N]
        H = x[3*N+1]

        # Initial radius estimate
        area = W * H
        r_avg = np.sqrt(area / N) * 0.5
        rs = np.clip(rs, 0.02, r_avg)

        for it in range(n_iter):
            temp = 1.0 / (it + 1) if cooling else 0.5
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.hypot(dx, dy)
                    min_d = rs[i] + rs[j]
                    if d < min_d and d > 1e-10:
                        overlap = min_d - d
                        # Force proportional to overlap
                        f = 2.0 * overlap
                        fx[i] += f * dx / d
                        fy[i] += f * dy / d
                        fx[j] -= f * dx / d
                        fy[j] -= f * dy / d
                    elif d <= 1e-10:
                        # Random push
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)

            # Wall repulsion
            for i in range(N):
                if xs[i] < rs[i]:
                    fx[i] += 2.0 * (rs[i] - xs[i])
                if xs[i] > W - rs[i]:
                    fx[i] -= 2.0 * (xs[i] - (W - rs[i]))
                if ys[i] < rs[i]:
                    fy[i] += 2.0 * (rs[i] - ys[i])
                if ys[i] > H - rs[i]:
                    fy[i] -= 2.0 * (ys[i] - (H - rs[i]))

            # Mild centering force
            cx, cy = W/2, H/2
            for i in range(N):
                fx[i] += 0.01 * (cx - xs[i])
                fy[i] += 0.01 * (cy - ys[i])

            # Update positions
            step = 0.05 * temp
            for i in range(N):
                xs[i] += step * fx[i]
                ys[i] += step * fy[i]
                # Clamp to bounds
                xs[i] = np.clip(xs[i], rs[i], W - rs[i])
                ys[i] = np.clip(ys[i], rs[i], H - rs[i])

            # Grow radii
            for i in range(N):
                r_bound = min(xs[i], W - xs[i], ys[i], H - ys[i])
                r_max = r_bound
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    r_candidate = d - rs[j]
                    if r_candidate < r_max:
                        r_max = r_candidate
                r_max = max(r_max, 1e-8)
                # Grow toward r_max gradually
                rs[i] = min(r_max, rs[i] + 0.005)

        x[:N] = xs
        x[N:2*N] = ys
        x[2*N:3*N] = rs
        return x

    def local_search_positions(x, step_sizes=[0.03, 0.015, 0.007], n_steps=8):
        """
        Move individual circles to find better packings after radius relaxation.
        """
        best_x = x.copy()
        best_score = get_score(best_x)
        if best_score == -np.inf:
            return best_x

        for step in step_sizes:
            for _ in range(n_steps):
                if time.time() > deadline - 5:
                    return best_x
                improved = False
                # Pick a random subset of circles to move
                indices = list(range(N))
                random.shuffle(indices)
                for i in indices:
                    if time.time() > deadline - 5:
                        return best_x
                    base_score = get_score(best_x)
                    if base_score == -np.inf:
                        break
                    W = best_x[3*N]
                    H = best_x[3*N+1]
                    r_i = best_x[2*N+i]
                    # Try 8 directions + stay
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            nx = best_x[i] + dx
                            ny = best_x[N+i] + dy
                            if nx < r_i or nx > W - r_i:
                                continue
                            if ny < r_i or ny > H - r_i:
                                continue
                            new_x = best_x.copy()
                            new_x[i] = nx
                            new_x[N+i] = ny
                            new_x = relax_radii(new_x, n_iter=10)
                            s = get_score(new_x)
                            if s > base_score + 1e-9:
                                best_x = new_x
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    break
        return best_x

    def run_slsqp(x0, maxiter=100):
        try:
            res = minimize(
                objective, x0, method='SLSQP',
                bounds=[(1e-5, None)] * (3*N) + [(0.4, 1.6), (0.4, 1.6)],
                constraints=constraints_def,
                options={'maxiter': maxiter, 'ftol': 1e-7}
            )
            return res.x
        except:
            return x0

    # Main loop
    attempts = 0
    max_attempts = 60

    # Pre-defined aspect ratios to explore
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.83, 1.3, 0.77, 1.4, 0.71, 1.5, 0.67, 1.6, 0.625, 1.75, 0.57]

    while time.time() < deadline and attempts < max_attempts:
        attempts += 1

        if attempts <= len(aspect_ratios):
            W = aspect_ratios[attempts - 1]
        else:
            W = np.random.uniform(0.55, 1.45)
        H = 2.0 - W
        if H < 0.4:
            H = 0.4
            W = 1.6
        if W < 0.4:
            W = 0.4
            H = 1.6

        # Choose initialization
        if attempts % 5 == 0:
            x0 = random_init(W, H)
        elif attempts % 5 == 1:
            x0 = grid_init(W, H, noise=0.02)
        elif attempts % 5 == 2:
            # Different grid layout
            if N % 3 == 0:
                x0 = grid_init(W, H, cols=3, rows=7, noise=0.01)
            else:
                x0 = grid_init(W, H, cols=4, rows=6, noise=0.01)
        elif attempts % 5 == 3:
            x0 = grid_init(W, H, cols=5, rows=5, noise=0.03)
        else:
            x0 = grid_init(W, H, cols=7, rows=3, noise=0.02)

        # Force relaxation first
        x_fr = force_relax(x0, n_iter=40)
        x_fr = relax_radii(x_fr, n_iter=20)
        score_fr = get_score(x_fr)
        if score_fr > best_score:
            best_score = score_fr
            best_result = x_fr.copy()

        # SLSQP refinement
        if time.time() < deadline - 15:
            x_sl = run_slsqp(x_fr, maxiter=80)
            x_sl = relax_radii(x_sl, n_iter=20)
            score_sl = get_score(x_sl)
            if score_sl > best_score:
                best_score = score_sl
                best_result = x_sl.copy()

            # Local search on the best
            if time.time() < deadline - 8 and score_sl > -np.inf:
                x_ls = local_search_positions(x_sl, step_sizes=[0.02, 0.01], n_steps=5)
                score_ls = get_score(x_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_result = x_ls.copy()

        # Also try SLSQP directly on grid init sometimes
        if attempts % 3 == 0 and time.time() < deadline - 20:
            x0b = grid_init(W, H, noise=0.05)
            x_sl2 = run_slsqp(x0b, maxiter=60)
            x_sl2 = relax_radii(x_sl2, n_iter=15)
            score_sl2 = get_score(x_sl2)
            if score_sl2 > best_score:
                best_score = score_sl2
                best_result = x_sl2.copy()

    # Final refinement on best result
    if best_result is not None and time.time() < deadline - 5:
        best_result = relax_radii(best_result, n_iter=30)
        s = get_score(best_result)
        if s > best_score:
            best_score = s

        if time.time() < deadline - 3:
            best_result = local_search_positions(best_result, step_sizes=[0.01, 0.005], n_steps=4)
            s = get_score(best_result)
            if s > best_score:
                best_score = s

    if best_result is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    result = np.column_stack((xs, ys, rs))
    return result
