import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Vectorized validity check (no sqrt)
    def is_valid(params, tol=1e-9):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist2 = dx*dx + dy*dy
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(dist2[iu] < (min_dist[iu] - tol)**2):
            return False
        return True

    def get_score(params):
        return np.sum(params[2*N:])

    # Clearance-based objective: minimize negative clearance
    # clearance = min over constraints of (distance - required)
    def clearance_objective(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        # Boundary clearances
        b = np.concatenate([
            xy[:, 0] - r,
            1 - xy[:, 0] - r,
            xy[:, 1] - r,
            1 - xy[:, 1] - r
        ])
        # Pairwise clearances (squared to avoid sqrt, but we need signed)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        p = dist[iu] - min_dist[iu]
        all_c = np.concatenate([b, p])
        # Minimize: -sum(r) + penalty for negative clearance
        # Use a smooth min approximation
        m = np.min(all_c)
        return -np.sum(r) + 1000.0 * np.maximum(0, -m)**2 + 10.0 * np.sum(np.maximum(0, -all_c))

    def softplus(x, beta=50):
        return np.log1p(np.exp(beta*x)) / beta

    def penalty_objective(params):
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        b = np.concatenate([
            xy[:, 0] - r,
            1 - xy[:, 0] - r,
            xy[:, 1] - r,
            1 - xy[:, 1] - r
        ])
        bv = np.sum(softplus(-b)**2)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        ov = np.sum(softplus(min_dist[iu] - dist[iu])**2)
        return -np.sum(r) + 100.0*bv + 100.0*ov

    def push_apart(params, n_iters=100):
        xy = params[:2*N].reshape(N, 2).copy()
        r = params[2*N:].copy()
        for _ in range(n_iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    md = r[i] + r[j]
                    if dist < md and dist > 1e-12:
                        ov = md - dist
                        wi = r[i]/(r[i]+r[j]) if r[i]+r[j] > 1e-15 else 0.5
                        wj = 1 - wi
                        px = (dx/dist)*ov
                        py = (dy/dist)*ov
                        xy[i, 0] -= px*wi
                        xy[i, 1] -= py*wi
                        xy[j, 0] += px*wj
                        xy[j, 1] += py*wj
                        moved = True
                    elif dist <= 1e-12:
                        ang = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01*np.cos(ang)
                        xy[i, 1] -= 0.01*np.sin(ang)
                        xy[j, 0] += 0.01*np.cos(ang)
                        xy[j, 1] += 0.01*np.sin(ang)
                        moved = True
            for i in range(N):
                if xy[i, 0] < r[i]: xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1-r[i]: xy[i, 0] = 1-r[i]; moved = True
                if xy[i, 1] < r[i]: xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1-r[i]: xy[i, 1] = 1-r[i]; moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def refine(params, obj, max_iter=400, tol=1e-14):
        try:
            res = minimize(obj, params, method='Powell',
                           options={'maxiter': max_iter, 'ftol': tol, 'maxfev': 3000})
            return res.x
        except Exception:
            return params

    def refine_lbfgs(params, max_iter=500, tol=1e-14):
        try:
            res = minimize(penalty_objective, params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50, 'maxcor': 20})
            return res.x
        except Exception:
            return params

    def initial_configs(rng):
        configs = []
        n = N
        # Grid
        side = 6
        for i in range(n):
            row, col = i//side, i%side
            configs.append([(col+0.5)/side, (row+0.5)/side, 0.5/side-0.001])
        cfg = np.array(configs)
        configs_list = [np.concatenate([cfg[:, :2].flatten(), cfg[:, 2]])]

        # Perturbed grid
        xy = cfg[:, :2].copy() + rng.uniform(-0.02, 0.02, size=(n, 2))
        r = cfg[:, 2].copy()
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Hexagonal
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//6, i%6
            x = (col+0.5)/6 + (0.25 if row % 2 == 1 else 0)
            y = (row+0.5)/6
            xy[i] = [x, y]; r[i] = 0.08
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random
        xy = rng.uniform(0.08, 0.92, size=(n, 2))
        r = rng.uniform(0.05, 0.1, size=n)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # 8x4
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//8, i%8
            xy[i] = [(col+0.5)/8, (row+0.5)/4]
            r[i] = 0.5/8 - 0.001
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # 4x8
        xy = np.zeros((n, 2)); r = np.zeros(n)
        for i in range(n):
            row, col = i//4, i%4
            xy[i] = [(col+0.5)/4, (row+0.5)/8]
            r[i] = 0.5/4 - 0.001
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random medium
        xy = rng.uniform(0.08, 0.92, size=(n, 2))
        r = rng.uniform(0.06, 0.09, size=n)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        # Random small
        xy = rng.uniform(0.05, 0.95, size=(n, 2))
        r = rng.uniform(0.03, 0.06, size=n)
        configs_list.append(np.concatenate([xy.flatten(), r]))

        return configs_list

    rng = np.random.RandomState(42)
    best_result = None
    best_score = -np.inf

    def update_best(params):
        nonlocal best_result, best_score
        if is_valid(params):
            s = get_score(params)
            if s > best_score:
                best_score = s
                best_result = params.copy()

    # Phase 1: diverse restarts
    base_configs = initial_configs(rng)
    n_restarts = 40
    for restart in range(n_restarts):
        if elapsed() > deadline - 40:
            break
        # Cycle through base configs with perturbation
        base = base_configs[restart % len(base_configs)]
        scale = 0.01 + 0.02 * (restart // len(base_configs))
        params = base.copy()
        xy_p = params[:2*N].reshape(N, 2).copy()
        r_p = params[2*N:].copy()
        xy_p += rng.uniform(-scale, scale, size=(N, 2))
        r_p += rng.uniform(-scale*0.3, scale*0.3, size=N)
        r_p = np.clip(r_p, 0.005, 0.5)
        xy_p = np.clip(xy_p, r_p[:, None], (1-r_p)[:, None])
        params = np.concatenate([xy_p.flatten(), r_p])
        params = push_apart(params, n_iters=40)
        update_best(params)

        # Powell on clearance objective
        polished = refine(params, clearance_objective, max_iter=300, tol=1e-13)
        update_best(polished)

        # L-BFGS-B on penalty
        polished2 = refine_lbfgs(polished, max_iter=300, tol=1e-13)
        update_best(polished2)

    # Phase 2: polish best
    if best_result is not None:
        polish_scales = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        for scale in polish_scales:
            if elapsed() > deadline - 5:
                break
            for trial in range(3):
                if elapsed() > deadline - 4:
                    break
                pert = best_result.copy()
                xy_p = pert[:2*N].reshape(N, 2).copy()
                r_p = pert[2*N:].copy()
                xy_p += rng.uniform(-scale, scale, size=(N, 2))
                r_p += rng.uniform(-scale*0.5, scale*0.5, size=N)
                r_p = np.clip(r_p, 0.001, 0.5)
                xy_p = np.clip(xy_p, r_p[:, None], (1-r_p)[:, None])
                pert = np.concatenate([xy_p.flatten(), r_p])
                pert = push_apart(pert, n_iters=50)
                polished = refine(pert, clearance_objective, max_iter=500, tol=1e-15)
                update_best(polished)
                polished2 = refine_lbfgs(polished, max_iter=500, tol=1e-15)
                update_best(polished2)

    if best_result is not None:
        xy = best_result[:2*N].reshape(N, 2).copy()
        r = best_result[2*N:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final = np.concatenate([xy.flatten(), r])
        if is_valid(final):
            return np.column_stack([xy, r])

    # Fallback grid
    side = 6
    cells = []
    for i in range(N):
        row, col = i//side, i%side
        cells.append([(col+0.5)/side, (row+0.5)/side, 0.5/side - 1e-3])
    return np.array(cells)
