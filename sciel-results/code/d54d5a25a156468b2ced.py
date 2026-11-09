import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def elapsed():
        return time.time() - start_time

    def validate(x, y, r, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def compute_optimal_radii(x, y, max_iter=50000, tol=1e-14):
        """Given fixed positions, find maximum feasible radii via Jacobi iteration."""
        # Distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)

        # Wall limits
        wall_limit = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        # Initialize radii to half the wall limit
        r = wall_limit * 0.5

        for it in range(max_iter):
            if time.time() > deadline:
                break
            new_r = r.copy()
            for i in range(N):
                cand = wall_limit[i]
                for j in range(N):
                    if i == j:
                        continue
                    dl = dist[i, j]
                    if dl < 1e-15:
                        continue
                    nl = dl - r[j]
                    if nl < cand:
                        cand = nl
                new_r[i] = max(0.0, cand)
            delta = np.max(np.abs(new_r - r))
            r = new_r
            if delta < tol:
                break
        return r

    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        # We maximize sum(r), so minimize -sum(r)
        # Add a small penalty for wall violations to keep circles inside
        # But primarily we want to maximize sum of radii
        # Since radii are determined by positions, we can just optimize positions
        # and compute optimal radii
        # However, for the optimizer, we need a smooth objective
        # Let's use the actual radii from the optimal computation
        r_opt = compute_optimal_radii(x, y, max_iter=2000)
        return -np.sum(r_opt)

    def grad_objective(params):
        # Numerical gradient since analytical is complex with the inner optimization
        eps = 1e-7
        g = np.zeros(3 * N)
        f0 = objective(params)
        for i in range(3 * N):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            g[i] = (f_plus - f0) / eps
        return g

    def optimize_positions(x0, y0, max_iter=1000):
        """Optimize positions to maximize sum of optimal radii."""
        params0 = np.concatenate([x0, y0, np.zeros(N)])

        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
        for i in range(N):
            bounds.append((0.01, 0.99))  # y
        for i in range(N):
            bounds.append((0.0, 0.0))  # r not optimized directly

        # Use a simpler approach: just optimize x and y
        def pos_objective(pos_params):
            x = pos_params[0:N]
            y = pos_params[N:2*N]
            r_opt = compute_optimal_radii(x, y, max_iter=2000)
            return -np.sum(r_opt)

        def pos_grad(pos_params):
            eps = 1e-7
            g = np.zeros(2 * N)
            f0 = pos_objective(pos_params)
            for i in range(2 * N):
                pos_plus = pos_params.copy()
                pos_plus[i] += eps
                f_plus = pos_objective(pos_plus)
                g[i] = (f_plus - f0) / eps
            return g

        result = minimize(
            pos_objective,
            pos_params,
            method='L-BFGS-B',
            jac=pos_grad,
            bounds=bounds[:2*N],
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x[:N], result.x[N:2*N]

    def local_search(x, y, max_iters=5000):
        """Local search: try small random perturbations to escape local minima."""
        best_x = x.copy()
        best_y = y.copy()
        best_r = compute_optimal_radii(best_x, best_y)
        best_score = np.sum(best_r)

        rng = np.random.RandomState(42)
        step_size = 0.01

        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Random perturbation
            dx = rng.normal(0, step_size, N)
            dy = rng.normal(0, step_size, N)
            new_x = np.clip(best_x + dx, 0.01, 0.99)
            new_y = np.clip(best_y + dy, 0.01, 0.99)

            new_r = compute_optimal_radii(new_x, new_y, max_iter=2000)
            new_score = np.sum(new_r)

            if new_score > best_score:
                best_x = new_x
                best_y = new_y
                best_r = new_r
                best_score = new_score
                # Reduce step size slightly
                step_size *= 0.99
            else:
                # Increase step size if stuck
                step_size = min(0.05, step_size * 1.05)

        return best_x, best_y, best_r

    def generate_initial_configs():
        """Generate diverse initial configurations."""
        configs = []

        # 1. Grid 5x5 + center
        side = 5
        spacing = 1.0 / side
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append(((i + 0.5) * spacing, (j + 0.5) * spacing))
        grid.append((0.5, 0.5))
        x = np.array([p[0] for p in grid])
        y = np.array([p[1] for p in grid])
        configs.append((x, y))

        # 2. Hexagonal packing
        hex_configs = []
        for row in range(5):
            for col in range(5):
                xx = (col + 0.5) * 0.2
                yy = (row + 0.5) * 0.2
                if row % 2 == 1:
                    xx += 0.1
                if xx < 1.0 and yy < 1.0:
                    hex_configs.append((xx, yy))
        hex_configs = hex_configs[:26]
        if len(hex_configs) == 26:
            x = np.array([p[0] for p in hex_configs])
            y = np.array([p[1] for p in hex_configs])
            configs.append((x, y))

        # 3. Random uniform
        for seed in range(5):
            rng = np.random.RandomState(100 + seed)
            x = rng.uniform(0.15, 0.85, N)
            y = rng.uniform(0.15, 0.85, N)
            configs.append((x, y))

        # 4. Clustered: two groups
        for seed in range(3):
            rng = np.random.RandomState(200 + seed)
            x1 = rng.uniform(0.1, 0.45, 13)
            y1 = rng.uniform(0.1, 0.9, 13)
            x2 = rng.uniform(0.55, 0.9, 13)
            y2 = rng.uniform(0.1, 0.9, 13)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            configs.append((x, y))

        # 5. Ring + center
        for seed in range(2):
            rng = np.random.RandomState(300 + seed)
            angles1 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x1 = 0.5 + 0.35 * np.cos(angles1)
            y1 = 0.5 + 0.35 * np.sin(angles1)
            angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x2 = 0.5 + 0.18 * np.cos(angles2)
            y2 = 0.5 + 0.18 * np.sin(angles2)
            x = np.concatenate([x1, x2, [0.5, 0.5]])
            y = np.concatenate([y1, y2, [0.5, 0.5]])
            configs.append((x, y))

        return configs

    # Generate initial configs
    configs = generate_initial_configs()

    best_score = 0.0
    best_config = None

    # Process each initial config
    for idx, (x0, y0) in enumerate(configs):
        if elapsed() > deadline - 10:
            break

        # First, solve radii for the initial positions
        r0 = compute_optimal_radii(x0, y0)
        if validate(x0, y0, r0):
            s = float(np.sum(r0))
            if s > best_score:
                best_score = s
                best_config = np.column_stack([x0, y0, r0])

        # Run local search
        try:
            x_ls, y_ls, r_ls = local_search(x0, y0, max_iters=2000)
            if validate(x_ls, y_ls, r_ls):
                s = float(np.sum(r_ls))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_ls, y_ls, r_ls])
        except Exception as e:
            pass

        # Run gradient-based optimization
        if elapsed() < deadline - 15:
            try:
                x_opt, y_opt = optimize_positions(x0, y0, max_iter=500)
                r_opt = compute_optimal_radii(x_opt, y_opt)
                if validate(x_opt, y_opt, r_opt):
                    s = float(np.sum(r_opt))
                    if s > best_score:
                        best_score = s
                        best_config = np.column_stack([x_opt, y_opt, r_opt])
            except Exception as e:
                pass

    # Final refinement pass
    if best_config is not None and elapsed() < deadline - 5:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()

        # More local search
        x_ls, y_ls, r_ls = local_search(x_cur, y_cur, max_iters=5000)
        if validate(x_ls, y_ls, r_ls):
            s = float(np.sum(r_ls))
            if s > best_score:
                best_score = s
                best_config = np.column_stack([x_ls, y_ls, r_ls])

    # Fallback if no valid config found
    if best_config is None:
        side_f = 6
        s = 1.0 / side_f
        cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
        best_config = np.array(cells[:N])
        best_score = float(np.sum(best_config[:, 2]))

    # Final validation
    x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
    if not validate(x_f, y_f, r_f):
        for factor in [0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7]:
            r_f2 = r_f * factor
            if validate(x_f, y_f, r_f2):
                best_config = np.column_stack([x_f, y_f, r_f2])
                break
        else:
            side_f = 6
            s = 1.0 / side_f
            cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
            best_config = np.array(cells[:N])

    return best_config
