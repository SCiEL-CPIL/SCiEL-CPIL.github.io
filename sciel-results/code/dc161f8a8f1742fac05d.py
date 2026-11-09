import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def objective_and_grad(params):
        """
        params: (x, y, r) flattened, length 3*N
        Objective: maximize sum(r) subject to:
          - circles inside [0,1]^2
          - non-overlapping
        We minimize: -sum(r) + penalty * sum of violations
        """
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: negative sum of radii (we minimize)
        obj = -np.sum(r)
        
        # Penalty for violations
        penalty = 0.0
        grad_obj = np.zeros(3 * N)
        
        # Wall constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Soft penalty: if x < r, add penalty * (r - x)^2
        # We use a quadratic penalty for smoothness
        
        # For each circle, check wall violations
        # x - r < 0 => violation
        # 1 - x - r < 0 => violation
        # y - r < 0 => violation
        # 1 - y - r < 0 => violation
        
        # We'll use a soft penalty: max(0, violation)^2
        # Gradient computed analytically
        
        # Wall: x_i - r_i
        viol_x_left = np.maximum(0, r - x)
        viol_x_right = np.maximum(0, r - (1 - x))
        viol_y_left = np.maximum(0, r - y)
        viol_y_right = np.maximum(0, r - (1 - y))
        
        penalty += np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + \
                   np.sum(viol_y_left**2) + np.sum(viol_y_right**2)
        
        # Gradients for wall violations
        # d/dx of (r-x)^2 when x<r: -2(r-x)
        # d/dr of (r-x)^2 when x<r: 2(r-x)
        grad_obj[0:N] += 2 * viol_x_left * (-1)  # d/dx
        grad_obj[N:2*N] += 2 * viol_y_left * (-1)  # d/dy
        grad_obj[2*N:3*N] += 2 * viol_x_left * (1)  # d/dr
        
        grad_obj[0:N] += 2 * viol_x_right * (1)  # d/dx of (r-1+x)^2 = 2(r-1+x)
        grad_obj[N:2*N] += 2 * viol_y_right * (1)
        grad_obj[2*N:3*N] += 2 * viol_x_right * (1)
        
        grad_obj[0:N] += 2 * viol_y_left * 0
        grad_obj[N:2*N] += 2 * viol_y_left * 1  # wait, let me redo this carefully
        
        # Let me recompute gradients more carefully
        # Reset gradient
        grad_obj = np.zeros(3 * N)
        # Base objective: -sum(r)
        grad_obj[2*N:3*N] = -1.0
        
        # Wall violations
        # f1_i = max(0, r_i - x_i)^2
        v1 = np.maximum(0, r - x)
        penalty += np.sum(v1**2)
        # d/dx_i: -2 * v1_i (when v1_i > 0)
        # d/dr_i: 2 * v1_i
        mask1 = v1 > 0
        grad_obj[0:N][mask1] += -2 * v1[mask1]
        grad_obj[2*N:3*N][mask1] += 2 * v1[mask1]
        
        # f2_i = max(0, r_i - (1 - x_i))^2 = max(0, r_i + x_i - 1)^2
        v2 = np.maximum(0, r + x - 1)
        penalty += np.sum(v2**2)
        mask2 = v2 > 0
        grad_obj[0:N][mask2] += 2 * v2[mask2]
        grad_obj[2*N:3*N][mask2] += 2 * v2[mask2]
        
        # f3_i = max(0, r_i - y_i)^2
        v3 = np.maximum(0, r - y)
        penalty += np.sum(v3**2)
        mask3 = v3 > 0
        grad_obj[N:2*N][mask3] += -2 * v3[mask3]
        grad_obj[2*N:3*N][mask3] += 2 * v3[mask3]
        
        # f4_i = max(0, r_i - (1 - y_i))^2 = max(0, r_i + y_i - 1)^2
        v4 = np.maximum(0, r + y - 1)
        penalty += np.sum(v4**2)
        mask4 = v4 > 0
        grad_obj[N:2*N][mask4] += 2 * v4[mask4]
        grad_obj[2*N:3*N][mask4] += 2 * v4[mask4]
        
        # Pairwise overlap: for i < j, max(0, r_i + r_j - d_ij)^2
        # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # To avoid sqrt issues, use d^2 form? No, let's use direct.
        # For efficiency, compute all pairs
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    # Degenerate: push apart
                    viol = r[i] + r[j]
                    penalty += viol**2
                    grad_obj[i] += 2 * viol * 0  # x_i
                    grad_obj[N+i] += 2 * viol * 0  # y_i
                    grad_obj[2*N+i] += 2 * viol  # r_i
                    grad_obj[j] += 2 * viol * 0
                    grad_obj[N+j] += 2 * viol * 0
                    grad_obj[2*N+j] += 2 * viol
                    continue
                viol = max(0, r[i] + r[j] - d)
                if viol > 0:
                    penalty += viol**2
                    # d/dx_i of viol^2 = 2*viol * d/dx_i(r_i+r_j-d) = 2*viol * (-dx/d)
                    # d/dr_i = 2*viol * 1
                    # d/dr_j = 2*viol * 1
                    # d/dx_j = 2*viol * (dx/d)
                    # d/dy_i = 2*viol * (-dy/d)
                    # d/dy_j = 2*viol * (dy/d)
                    grad_obj[i] += 2 * viol * (-dx / d)
                    grad_obj[N+i] += 2 * viol * (-dy / d)
                    grad_obj[2*N+i] += 2 * viol
                    grad_obj[j] += 2 * viol * (dx / d)
                    grad_obj[N+j] += 2 * viol * (dy / d)
                    grad_obj[2*N+j] += 2 * viol
        
        total = obj + 1000.0 * penalty
        return total, grad_obj + 1000.0 * (grad_obj - grad_obj)  # placeholder, fix below

    # Let me rewrite the objective more cleanly
    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        obj = -np.sum(r)
        penalty = 0.0
        # Wall
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r + x - 1)
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r + y - 1)
        penalty += np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                d = np.sqrt(dx*dx+dy*dy)
                if d < 1e-12:
                    v = r[i]+r[j]
                else:
                    v = max(0, r[i]+r[j]-d)
                penalty += v**2
        return obj + 1000.0 * penalty

    def grad_objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        g = np.zeros(3*N)
        g[2*N:3*N] = -1.0  # d(-sum r)/dr = -1
        # Wall
        v1 = np.maximum(0, r - x)
        m1 = v1 > 0
        g[0:N][m1] += 1000.0 * (-2 * v1[m1])
        g[2*N:3*N][m1] += 1000.0 * (2 * v1[m1])
        v2 = np.maximum(0, r + x - 1)
        m2 = v2 > 0
        g[0:N][m2] += 1000.0 * (2 * v2[m2])
        g[2*N:3*N][m2] += 1000.0 * (2 * v2[m2])
        v3 = np.maximum(0, r - y)
        m3 = v3 > 0
        g[N:2*N][m3] += 1000.0 * (-2 * v3[m3])
        g[2*N:3*N][m3] += 1000.0 * (2 * v3[m3])
        v4 = np.maximum(0, r + y - 1)
        m4 = v4 > 0
        g[N:2*N][m4] += 1000.0 * (2 * v4[m4])
        g[2*N:3*N][m4] += 1000.0 * (2 * v4[m4])
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                d = np.sqrt(dx*dx+dy*dy)
                if d < 1e-12:
                    v = r[i]+r[j]
                    g[2*N+i] += 1000.0 * 2 * v
                    g[2*N+j] += 1000.0 * 2 * v
                    continue
                v = max(0, r[i]+r[j]-d)
                if v <= 0:
                    continue
                g[i] += 1000.0 * 2 * v * (-dx/d)
                g[N+i] += 1000.0 * 2 * v * (-dy/d)
                g[2*N+i] += 1000.0 * 2 * v
                g[j] += 1000.0 * 2 * v * (dx/d)
                g[N+j] += 1000.0 * 2 * v * (dy/d)
                g[2*N+j] += 1000.0 * 2 * v
        return g

    def validate(x, y, r, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]
                dy = y[i]-y[j]
                d = np.sqrt(dx*dx+dy*dy)
                if d < r[i]+r[j] - tol:
                    return False
        return True

    def make_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        # 1. Grid 5x5 + center (like before but as one of many starts)
        side = 5
        spacing = 1.0/side
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append(((i+0.5)*spacing, (j+0.5)*spacing))
        grid.append((0.5, 0.5))
        x = np.array([p[0] for p in grid])
        y = np.array([p[1] for p in grid])
        r = np.full(N, 0.08)
        configs.append(np.concatenate([x, y, r]))
        
        # 2. Hexagonal-like packing
        # 4 rows of circles
        # Row lengths: 7,6,7,6 or similar
        # Let's do a hex packing
        hex_configs = []
        # 5 rows of 5, offset
        for row in range(5):
            for col in range(5):
                xx = (col + 0.5) * 0.2
                yy = (row + 0.5) * 0.2
                if row % 2 == 1:
                    xx += 0.1
                if xx < 1.0 and yy < 1.0:
                    hex_configs.append((xx, yy))
        # Take first 26
        hex_configs = hex_configs[:26]
        if len(hex_configs) == 26:
            x = np.array([p[0] for p in hex_configs])
            y = np.array([p[1] for p in hex_configs])
            r = np.full(N, 0.08)
            configs.append(np.concatenate([x, y, r]))
        
        # 3. Random uniform
        rng = np.random.RandomState(42)
        for seed in range(5):
            rng = np.random.RandomState(100 + seed)
            x = rng.uniform(0.15, 0.85, N)
            y = rng.uniform(0.15, 0.85, N)
            r = np.full(N, 0.07)
            configs.append(np.concatenate([x, y, r]))
        
        # 4. Clustered: two groups
        for seed in range(3):
            rng = np.random.RandomState(200 + seed)
            # Left half and right half
            x1 = rng.uniform(0.1, 0.45, 13)
            y1 = rng.uniform(0.1, 0.9, 13)
            x2 = rng.uniform(0.55, 0.9, 13)
            y2 = rng.uniform(0.1, 0.9, 13)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            r = np.full(N, 0.07)
            configs.append(np.concatenate([x, y, r]))
        
        # 5. Ring + center
        # 12 on a ring of radius 0.35, 12 on ring of radius 0.18, 2 center
        for seed in range(2):
            rng = np.random.RandomState(300 + seed)
            angles1 = np.linspace(0, 2*np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x1 = 0.5 + 0.35 * np.cos(angles1)
            y1 = 0.5 + 0.35 * np.sin(angles1)
            angles2 = np.linspace(0, 2*np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x2 = 0.5 + 0.18 * np.cos(angles2)
            y2 = 0.5 + 0.18 * np.sin(angles2)
            x = np.concatenate([x1, x2, [0.5, 0.5]])
            y = np.concatenate([y1, y2, [0.5, 0.5]])
            r = np.full(N, 0.07)
            configs.append(np.concatenate([x, y, r]))
        
        return configs

    def optimize_from_start(params0, time_limit):
        """Run L-BFGS-B from initial params."""
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
        for i in range(N):
            bounds.append((0.0, 1.0))  # y
        for i in range(N):
            bounds.append((0.0, 0.5))  # r
        
        start_t = time.time()
        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            jac=grad_objective,
            bounds=bounds,
            options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result

    def solve_radii_for_positions(x, y, max_iter=30000, tol=1e-12):
        """Given fixed positions, solve for optimal radii via iteration."""
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        wall_limit = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
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
                    if dl < 1e-9:
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

    # Generate initial configs
    configs = make_initial_configs()
    
    best_score = 0.0
    best_config = None
    
    # Process each initial config
    for idx, cfg in enumerate(configs):
        if elapsed() > deadline - 5:
            break
        
        # First, solve radii for the initial positions to get a good starting point
        x0 = cfg[0:N]
        y0 = cfg[N:2*N]
        r0 = solve_radii_for_positions(x0, y0)
        params0 = np.concatenate([x0, y0, r0])
        
        # Check if initial is valid
        if validate(x0, y0, r0):
            s = float(np.sum(r0))
            if s > best_score:
                best_score = s
                best_config = np.column_stack([x0, y0, r0])
        
        # Run L-BFGS-B
        try:
            result = optimize_from_start(params0, deadline - elapsed())
            params_opt = result.x
            x_opt = params_opt[0:N]
            y_opt = params_opt[N:2*N]
            r_opt = params_opt[2*N:3*N]
            
            # Validate and possibly repair
            if validate(x_opt, y_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_opt, y_opt, r_opt])
            else:
                # Try to repair by solving radii for the optimized positions
                r_repair = solve_radii_for_positions(x_opt, y_opt)
                if validate(x_opt, y_opt, r_repair):
                    s = float(np.sum(r_repair))
                    if s > best_score:
                        best_score = s
                        best_config = np.column_stack([x_opt, y_opt, r_repair])
        except Exception as e:
            pass
    
    # If we have a good config, do a final refinement pass with more iterations
    if best_config is not None and elapsed() < deadline - 10:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()
        
        # Refine with L-BFGS-B from current best
        params_cur = np.concatenate([x_cur, y_cur, r_cur])
        try:
            result = optimize_from_start(params_cur, deadline - elapsed())
            params_opt = result.x
            x_opt = params_opt[0:N]
            y_opt = params_opt[N:2*N]
            r_opt = params_opt[2*N:3*N]
            if validate(x_opt, y_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_opt, y_opt, r_opt])
            else:
                r_repair = solve_radii_for_positions(x_opt, y_opt)
                if validate(x_opt, y_opt, r_repair):
                    s = float(np.sum(r_repair))
                    if s > best_score:
                        best_score = s
                        best_config = np.column_stack([x_opt, y_opt, r_repair])
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
