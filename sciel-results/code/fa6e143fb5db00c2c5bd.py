"""
Circle packing in unit square using contact-graph constrained optimization.
Circles are pushed against each other and boundaries via SLSQP.
Uses radial-sweep initialization for outer ring and multi-start SLSQP.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def check_validity(x, y, r):
        """Check if circles are valid (inside unit square, non-overlapping)."""
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def _score(x, y, r):
        """Compute the objective value (sum of radii) if valid, else -inf."""
        if check_validity(x, y, r):
            return np.sum(r)
        for scale in [0.999, 0.99, 0.98, 0.95, 0.90]:
            r_test = r * scale
            if check_validity(x, y, r_test):
                return np.sum(r_test)
        return -np.inf

    def radial_sweep_init(seed):
        """
        Radial-sweep initialization: central cluster + outer ring.
        Outer circles are placed on a ring around center with optimized angles.
        """
        rng = np.random.RandomState(seed)
        
        # Central cluster: 10 circles in a tight inner arrangement
        n_center = 10
        n_outer = N - n_center  # 16 outer circles
        
        # Center cluster: hexagonal packing around (0.5, 0.5)
        cx, cy = 0.5, 0.5
        inner_r = 0.08
        x_center = np.zeros(n_center)
        y_center = np.zeros(n_center)
        
        # First circle at center
        x_center[0] = cx
        y_center[0] = cy
        
        # Ring of 6 around center
        for k in range(1, 7):
            angle = 2 * np.pi * (k - 1) / 6
            x_center[k] = cx + inner_r * np.cos(angle)
            y_center[k] = cy + inner_r * np.sin(angle)
        
        # Second partial ring
        for k in range(7, 10):
            angle = 2 * np.pi * (k - 7) / 3 + np.pi / 3
            x_center[k] = cx + 2 * inner_r * np.cos(angle)
            y_center[k] = cy + 2 * inner_r * np.sin(angle)
        
        # Outer ring: 16 circles on a ring of radius ~0.35
        outer_radius = 0.35 + rng.uniform(-0.03, 0.03)
        angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi/n_outer)
        
        x_outer = cx + outer_radius * np.cos(angles)
        y_outer = cy + outer_radius * np.sin(angles)
        
        # Clip to stay in bounds
        x_outer = np.clip(x_outer, 0.05, 0.95)
        y_outer = np.clip(y_outer, 0.05, 0.95)
        
        x = np.concatenate([x_center, x_outer])
        y = np.concatenate([y_center, y_outer])
        
        x = np.clip(x, 0.03, 0.97)
        y = np.clip(y, 0.03, 0.97)
        return x, y

    def grid_init(seed, n_grid=5):
        """Grid-based initialization with jitter."""
        rng = np.random.RandomState(seed)
        side = n_grid
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        x += rng.uniform(-0.05, 0.05, N)
        y += rng.uniform(-0.05, 0.05, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def repulsion_relax_init(seed):
        """
        Repulsion-relaxation initialization: start with random positions,
        then iteratively push circles apart to reduce overlaps.
        """
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.15, 0.85, N)
        y = rng.uniform(0.15, 0.85, N)
        
        # Iterative repulsion relaxation
        for it in range(200):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Compute repulsion forces
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, 1.0)
            
            # Desired minimum separation
            min_sep = 0.08
            force_x = np.zeros(N)
            force_y = np.zeros(N)
            
            for i in range(N):
                for j in range(i+1, N):
                    d = dist[i, j]
                    if d < min_sep:
                        # Repulsive force proportional to overlap
                        overlap = min_sep - d
                        fx = dx[i, j] / d * overlap
                        fy = dy[i, j] / d * overlap
                        force_x[i] += fx
                        force_y[i] += fy
                        force_x[j] -= fx
                        force_y[j] -= fy
            
            # Boundary repulsion
            for i in range(N):
                if x[i] < 0.08:
                    force_x[i] += (0.08 - x[i]) * 2
                elif x[i] > 0.92:
                    force_x[i] -= (x[i] - 0.92) * 2
                if y[i] < 0.08:
                    force_y[i] += (0.08 - y[i]) * 2
                elif y[i] > 0.92:
                    force_y[i] -= (y[i] - 0.92) * 2
            
            # Apply forces with damping
            lr = 0.1 / (1 + it * 0.01)
            x += lr * force_x
            y += lr * force_y
            
            # Clip to bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        
        return x, y

    def slsqp_optimize(x0, y0, seed, max_iter=500):
        """
        SLSQP optimization with explicit non-overlap and boundary constraints.
        Variables: [x_0,...,x_{N-1}, y_0,...,y_{N-1}]
        Objective: maximize sum of radii (equivalently, minimize -sum of radii)
        Constraints: pairwise distance >= r_i + r_j, and boundary constraints.
        
        We use a penalty-augmented approach: maximize sum of radii subject to
        soft constraints, then verify.
        """
        rng = np.random.RandomState(seed)
        
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        # Build constraint functions for SLSQP
        # For each pair (i,j) with i<j: dist(i,j) - 0.5*(dist(i,j) + dist(i,j)) >= 0
        # Actually, the constraint is: dist(i,j) >= r_i + r_j
        # But r_i depends on position... This is implicit.
        
        # Alternative: use penalty method with explicit constraints
        # Constraint: x_i + r_i <= 1, x_i - r_i >= 0, etc.
        # And: dist(i,j) >= r_i + r_j
        
        # Since r depends on position, we use a two-level approach:
        # Level 1: optimize positions to maximize sum of min-distances
        # Level 2: compute radii
        
        def objective_and_grad(params):
            x = params[:N]
            y = params[N:2*N]
            
            # Compute radii
            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-30)
            np.fill_diagonal(dist, 1e10)
            
            min_dist = np.min(dist, axis=1)
            r = np.minimum(r_bound, 0.5 * min_dist)
            r = np.maximum(r, 1e-8)
            
            obj = -np.sum(r)
            
            # Gradient via finite differences approximation (vectorized)
            # Actually, let's use analytic gradient
            grad = np.zeros(2 * N)
            
            # For each circle i, dr_i/dx_k for all k
            # r_i = min(r_bound_i, 0.5 * min_j dist(i,j))
            # Use smooth approximation with soft-min
            
            beta = 80.0
            c = 0.5 * dist
            np.fill_diagonal(c, r_bound)
            
            neg_beta_c = -beta * c
            max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
            exp_terms = np.exp(neg_beta_c - max_neg)
            sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
            w = exp_terms / sum_exp  # (N, N)
            
            dist_safe = dist.copy()
            np.fill_diagonal(dist_safe, 1.0)
            
            # Gradient of 0.5*dist(i,j) w.r.t. x_i
            dc_dx_i = 0.5 * dx / dist_safe
            dc_dy_i = 0.5 * dy / dist_safe
            dc_dx_j = -0.5 * dx / dist_safe
            dc_dy_j = -0.5 * dy / dist_safe
            
            # Boundary gradient (soft)
            bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
            beta_b = 150.0
            neg_bb = -beta_b * bounds_4
            max_bb = np.max(neg_bb, axis=1, keepdims=True)
            exp_bb = np.exp(neg_bb - max_bb)
            sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
            w_b = exp_bb / sum_bb
            
            grad_rb_dx = w_b[:, 0] - w_b[:, 1]
            grad_rb_dy = w_b[:, 2] - w_b[:, 3]
            
            np.fill_diagonal(dc_dx_i, grad_rb_dx)
            np.fill_diagonal(dc_dy_i, grad_rb_dy)
            np.fill_diagonal(dc_dx_j, grad_rb_dx)
            np.fill_diagonal(dc_dy_j, grad_rb_dy)
            
            dr_dx_self = np.sum(w * dc_dx_i, axis=1)
            dr_dy_self = np.sum(w * dc_dy_i, axis=1)
            dr_dx_cross = np.sum(w * dc_dx_j, axis=0)
            dr_dy_cross = np.sum(w * dc_dy_j, axis=0)
            
            dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
            dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)
            
            grad[:N] = -dr_dx
            grad[N:] = -dr_dy
            
            return obj, grad
        
        # Run SLSQP with penalty for boundary violations
        def penalty_objective(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            
            # Penalty for boundary violations
            pen = 0.0
            pen += 100.0 * np.sum(np.maximum(0, x - 1 + r)**2)
            pen += 100.0 * np.sum(np.maximum(0, r - x)**2)
            pen += 100.0 * np.sum(np.maximum(0, y - 1 + r)**2)
            pen += 100.0 * np.sum(np.maximum(0, r - y)**2)
            
            return -np.sum(r) + pen
        
        # Use L-BFGS-B as it handles bounds well
        try:
            res = minimize(objective_and_grad, params0,
                          method='L-BFGS-B', bounds=bounds,
                          jac=True,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            params_opt = res.x
        except Exception:
            params_opt = params0
        
        x_opt = params_opt[:N]
        y_opt = params_opt[N:]
        r_opt = compute_radii(x_opt, y_opt)
        
        # Verify and scale if needed
        if not check_validity(x_opt, y_opt, r_opt):
            for scale in [0.999, 0.99, 0.98, 0.95]:
                r_test = r_opt * scale
                if check_validity(x_opt, y_opt, r_test):
                    r_opt = r_test
                    break
            else:
                r_opt = compute_radii(x_opt, y_opt)
        
        return x_opt, y_opt, r_opt, np.sum(r_opt)

    def _grow_radii(x, y, r, max_iter=50, tol=1e-10):
        """
        Iteratively grow radii while maintaining validity.
        For each circle, try to increase its radius slightly and check validity.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = len(r)
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            for i in range(n):
                # Try to grow radius i by small amount
                dr = 1e-5
                r_test = r.copy()
                r_test[i] += dr
                
                if check_validity(x, y, r_test):
                    r = r_test
                    improved = True
                else:
                    # Try smaller increment
                    for scale in [0.5, 0.25, 0.1]:
                        r_test[i] = r[i] + dr * scale
                        if check_validity(x, y, r_test):
                            r = r_test
                            improved = True
                            break
                        r_test[i] = r[i]
            
            if not improved:
                break
        
        return r

    def local_search_polish(x, y, max_rounds=30):
        """Local search with adaptive step sizes and radius growth."""
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])
        
        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)
        
        # Initial radius growth
        r = _grow_radii(x, y, r, max_iter=20)
        curr_val = np.sum(r)
        
        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            best_imp = 1e-10
            best_move = None
            found_any = False
            
            for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                for i in range(N):
                    for d in range(8):
                        nx = x[i] + dirs[d, 0] * step
                        ny = y[i] + dirs[d, 1] * step
                        if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
                            continue
                        
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = nx
                        y_temp[i] = ny
                        
                        r_temp = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_temp)
                        imp = new_val - curr_val
                        
                        if imp > best_imp:
                            best_imp = imp
                            best_move = (i, d, step, nx, ny)
                            found_any = True
                
                if best_imp > step * 0.05:
                    break
            
            if best_move is not None:
                i, d, step, nx, ny = best_move
                x[i] = nx
                y[i] = ny
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                # Try radius growth when no position move helps
                r_new = _grow_radii(x, y, r, max_iter=10)
                if np.sum(r_new) > curr_val + 1e-12:
                    r = r_new
                    curr_val = np.sum(r)
                else:
                    break
        
        r = _grow_radii(x, y, r, max_iter=30)
        r = compute_radii(x, y)
        return x, y, r, np.sum(r)

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start: radial sweep + grid + random + repulsion initializations
    # Each start: SLSQP/L-BFGS optimization + local search polish
    num_starts = 14
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 15:
            break

        # Choose initialization strategy
        if i % 4 == 0:
            x0, y0 = radial_sweep_init(i)
        elif i % 4 == 1:
            x0, y0 = grid_init(i, n_grid=5)
        elif i % 4 == 2:
            x0, y0 = random_init(i)
        else:
            x0, y0 = repulsion_relax_init(i)

        # Run gradient-based optimization
        x_opt, y_opt, r_opt, val_opt = slsqp_optimize(x0, y0, seed=i)

        # Local search polish (includes radius growth)
        if time.time() - start_time < DEADLINE - 10:
            x_ls, y_ls, r_ls, val_ls = local_search_polish(x_opt, y_opt, max_rounds=25)
            if val_ls > val_opt:
                x_opt, y_opt, r_opt, val_opt = x_ls, y_ls, r_ls, val_ls

        # Update best
        if val_opt > best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            best_y = y_opt.copy()
            best_r = r_opt.copy()

        # For some starts, also try a second optimization pass from the polished result
        if i % 3 == 1 and time.time() - start_time < DEADLINE - 10:
            x_opt2, y_opt2, r_opt2, val_opt2 = slsqp_optimize(x_opt, y_opt, seed=i + 100)
            if val_opt2 > val_opt:
                x_opt, y_opt, r_opt, val_opt = x_opt2, y_opt2, r_opt2, val_opt2
            if time.time() - start_time < DEADLINE - 8:
                x_ls2, y_ls2, r_ls2, val_ls2 = local_search_polish(x_opt, y_opt, max_rounds=15)
                if val_ls2 > val_opt:
                    x_opt, y_opt, r_opt, val_opt = x_ls2, y_ls2, r_ls2, val_ls2
            if val_opt > best_val:
                best_val = val_opt
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()

    # Final polish on the best solution
    if time.time() - start_time < DEADLINE - 5:
        # Local search with radius growth
        x_ls_f, y_ls_f, r_ls_f, val_ls_f = local_search_polish(best_x, best_y, max_rounds=35)
        if val_ls_f > best_val:
            best_val = val_ls_f
            best_x = x_ls_f
            best_y = y_ls_f
            best_r = r_ls_f

        # Another optimization pass
        if time.time() - start_time < DEADLINE - 5:
            x_opt_f, y_opt_f, r_opt_f, val_opt_f = slsqp_optimize(best_x, best_y, seed=999)
            if val_opt_f > best_val:
                best_val = val_opt_f
                best_x = x_opt_f
                best_y = y_opt_f
                best_r = r_opt_f
            # Polish after re-optimization
            if time.time() - start_time < DEADLINE - 4:
                x_ls_ff, y_ls_ff, r_ls_ff, val_ls_ff = local_search_polish(best_x, best_y, max_rounds=25)
                if val_ls_ff > best_val:
                    best_val = val_ls_ff
                    best_x = x_ls_ff
                    best_y = y_ls_ff
                    best_r = r_ls_ff

        # Final radius growth
        if time.time() - start_time < DEADLINE - 3:
            best_r = _grow_radii(best_x, best_y, best_r, max_iter=50)
            best_val = np.sum(best_r)

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
