import numpy as np
import time
from scipy.optimize import minimize
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

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
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        wall_limit = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
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

    def objective(params, n):
        """
        Objective function: minimize negative sum of radii + penalty for violations.
        params: array of shape (3n,) -> [x1,y1,r1, x2,y2,r2, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for circles going outside [0,1]^2
        penalty_boundary = 0.0
        for i in range(n):
            if x[i] < r[i]:
                penalty_boundary += (r[i] - x[i]) ** 2
            if x[i] > 1.0 - r[i]:
                penalty_boundary += (x[i] - (1.0 - r[i])) ** 2
            if y[i] < r[i]:
                penalty_boundary += (r[i] - y[i]) ** 2
            if y[i] > 1.0 - r[i]:
                penalty_boundary += (y[i] - (1.0 - r[i])) ** 2
        
        # Penalty for overlapping circles
        penalty_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty_overlap += (min_dist - dist) ** 2
        
        # Sum of radii (we want to maximize, so minimize negative)
        sum_r = np.sum(r)
        
        # Large penalty weight to enforce constraints
        penalty_weight = 100.0
        return -sum_r + penalty_weight * (penalty_boundary + penalty_overlap)

    def objective_grad(params, n):
        """
        Gradient of the objective function.
        """
        x = params[0::3].copy()
        y = params[1::3].copy()
        r = params[2::3].copy()
        
        grad = np.zeros_like(params)
        
        # Boundary penalties
        penalty_weight = 100.0
        for i in range(n):
            # x boundaries
            if x[i] < r[i]:
                grad[3*i] += 2 * penalty_weight * (r[i] - x[i])
                grad[3*i+2] -= 2 * penalty_weight * (r[i] - x[i])
            if x[i] > 1.0 - r[i]:
                grad[3*i] -= 2 * penalty_weight * (x[i] - (1.0 - r[i]))
                grad[3*i+2] += 2 * penalty_weight * (x[i] - (1.0 - r[i]))
            
            # y boundaries
            if y[i] < r[i]:
                grad[3*i+1] += 2 * penalty_weight * (r[i] - y[i])
                grad[3*i+2] -= 2 * penalty_weight * (r[i] - y[i])
            if y[i] > 1.0 - r[i]:
                grad[3*i+1] -= 2 * penalty_weight * (y[i] - (1.0 - r[i]))
                grad[3*i+2] += 2 * penalty_weight * (y[i] - (1.0 - r[i]))
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    diff = min_dist - dist
                    factor = 2 * penalty_weight * diff / dist
                    grad[3*i] += factor * dx
                    grad[3*i+1] += factor * dy
                    grad[3*i+2] += 2 * penalty_weight * diff
                    grad[3*j] -= factor * dx
                    grad[3*j+1] -= factor * dy
                    grad[3*j+2] += 2 * penalty_weight * diff
                elif dist <= 1e-10:
                    grad[3*i+2] += 2 * penalty_weight * min_dist
                    grad[3*j+2] += 2 * penalty_weight * min_dist
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        return grad

    def project_to_valid(circles):
        """Project circles to be inside unit square with positive radii."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            r = max(r, 1e-6)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            circles[i] = [x, y, r]
        return circles

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
                step_size *= 0.99
            else:
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

    def optimize_full(params0, max_iter=300):
        """Run L-BFGS-B on full (x, y, r) parameters."""
        try:
            result = minimize(
                objective,
                params0,
                args=(N,),
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x
        except Exception:
            return params0

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

        # Run local search on positions
        try:
            x_ls, y_ls, r_ls = local_search(x0, y0, max_iters=2000)
            if validate(x_ls, y_ls, r_ls):
                s = float(np.sum(r_ls))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_ls, y_ls, r_ls])
        except Exception:
            pass

        # Run full optimization on (x, y, r)
        if elapsed() < deadline - 15:
            try:
                # Interleave x, y, r
                params0 = np.zeros(3 * N)
                params0[0::3] = x0
                params0[1::3] = y0
                params0[2::3] = r0
                
                params_opt = optimize_full(params0, max_iter=300)
                x_opt = params_opt[0::3]
                y_opt = params_opt[1::3]
                r_opt = params_opt[2::3]
                
                # Project and validate
                circles = np.column_stack([x_opt, y_opt, r_opt])
                circles = project_to_valid(circles)
                
                if validate(circles[:, 0], circles[:, 1], circles[:, 2]):
                    s = float(np.sum(circles[:, 2]))
                    if s > best_score:
                        best_score = s
                        best_config = circles
            except Exception:
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

        # Final full optimization
        if elapsed() < deadline - 2:
            try:
                params0 = np.zeros(3 * N)
                params0[0::3] = best_config[:, 0]
                params0[1::3] = best_config[:, 1]
                params0[2::3] = best_config[:, 2]
                
                params_opt = optimize_full(params0, max_iter=200)
                x_opt = params_opt[0::3]
                y_opt = params_opt[1::3]
                r_opt = params_opt[2::3]
                
                circles = np.column_stack([x_opt, y_opt, r_opt])
                circles = project_to_valid(circles)
                
                if validate(circles[:, 0], circles[:, 1], circles[:, 2]):
                    s = float(np.sum(circles[:, 2]))
                    if s > best_score:
                        best_score = s
                        best_config = circles
            except Exception:
                pass

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
