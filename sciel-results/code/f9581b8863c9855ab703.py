"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Discrete best-fit local search for deterministic neighborhood improvement.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 112.0  # Safety margin from 120s limit


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

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min.
        beta controls sharpness: large beta -> hard min, small beta -> smooth average.
        Returns (radii, soft_min_values) where soft_min_values are the unclamped values.
        """
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row
        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r, soft_min_all

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses analytic gradient of the log-sum-exp soft-min.
        """
        x = params[:N]
        y = params[N:2*N]

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        # Set diagonal to a large value so it doesn't contribute
        np.fill_diagonal(dist, 1e10)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        # where c_ij are the constraints for circle i
        neg_beta_c = -beta * full_constraints
        max_nb = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_nb)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        r = -(max_nb + log_sum) / beta  # (N, 1)
        r = r[:, 0]  # (N,)

        # Gradient: dr_i/dp_k = sum_j [ (beta * exp(-beta*c_ij) / sum_j' exp(-beta*c_ij)) * dc_ij/dp_k ]
        # = sum_j [ w_ij * dc_ij/dp_k ]  where w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij)
        # Note: dr_i/dp = -1/beta * d/dp [log(sum exp(-beta c))] = sum_j w_ij * dc_ij/dp
        # Actually: d/dp log(sum_j exp(-beta c_ij)) = sum_j exp(-beta c_ij) * (-beta dc_ij/dp) / sum_j' exp(-beta c_ij')
        # So dr_i/dp = -(1/beta) * [sum_j exp(-beta c_ij) * (-beta dc_ij/dp) / sum_j' exp(-beta c_ij')]
        #           = sum_j [exp(-beta c_ij) / sum_j' exp(-beta c_ij')] * dc_ij/dp
        #           = sum_j w_ij * dc_ij/dp

        # Weights w_ij (N, N)
        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        # Gradient of half_dist[i,j] w.r.t. x_i, y_i, x_j, y_j
        # d(0.5*dist_ij)/dx_i = 0.5 * (x_i - x_j) / dist_ij = 0.5 * dx[i,j] / dist[i,j]
        # d(0.5*dist_ij)/dy_i = 0.5 * dy[i,j] / dist[i,j]
        # d(0.5*dist_ij)/dx_j = 0.5 * (x_j - x_i) / dist_ij = -0.5 * dx[i,j] / dist[i,j]
        # d(0.5*dist_ij)/dy_j = -0.5 * dy[i,j] / dist[i,j]

        # For boundary: r_bound[i] = min(x_i, 1-x_i, y_i, 1-y_i)
        # Gradient of r_bound[i] w.r.t. x_i: +1 if x_i < 1-x_i and x_i < y_i and x_i < 1-y_i, else -1 if 1-x_i is min, etc.
        # For soft-min, we handle this through the constraint matrix directly.
        # c_ii = r_bound[i], so dc_ii/dx_i depends on which boundary is active.
        # In soft-min, all contribute with weights, so:
        # dc_ii/dx_i = +1 if x_i is the "soft" min component... 
        # Actually r_bound = min(x, 1-x, y, 1-y). Its gradient is piecewise.
        # For the soft-min over [half_dist_ij for j!=i, r_bound_i], the constraint c_ii = r_bound_i.
        # dr_bound_i/dx_i: if x_i <= 1-x_i, x_i <= y_i, x_i <= 1-y_i: +1
        #                  if 1-x_i < x_i, 1-x_i <= y_i, 1-x_i <= 1-y_i: -1
        #                  etc.
        # For smoothness, we can use: the gradient of min is the gradient of the active component.
        # In soft-min, it's a weighted average. But since we're using soft-min over the full constraint set,
        # the diagonal entry's gradient w.r.t. x_i is:
        # d r_bound_i / dx_i = sign based on which of {x_i, 1-x_i, y_i, 1-y_i} is smallest.
        # Let's compute it as:
        # grad_x_rb[i] = +1 if x_i is min, -1 if 1-x_i is min, 0 otherwise
        # grad_y_rb[i] = +1 if y_i is min, -1 if 1-y_i is min, 0 otherwise

        # Compute gradients of constraints w.r.t. each parameter
        # For off-diagonal (i != j): dc_ij/dx_i = 0.5*dx[i,j]/dist[i,j], dc_ij/dy_i = 0.5*dy[i,j]/dist[i,j]
        #                            dc_ij/dx_j = -0.5*dx[i,j]/dist[i,j], dc_ij/dy_j = -0.5*dy[i,j]/dist[i,j]
        
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)  # avoid div by zero, but w on diagonal will handle
        # Actually for off-diagonal, dist is fine. For diagonal, we'll set gradient separately.
        
        d_c_dx_i = 0.5 * dx / dist_safe  # (N,N): d c_ij / dx_i for each (i,j)
        d_c_dy_i = 0.5 * dy / dist_safe
        d_c_dx_j = -0.5 * dx / dist_safe  # d c_ij / dx_j
        d_c_dy_j = -0.5 * dy / dist_safe

        # For diagonal: c_ii = r_bound[i]
        # d r_bound_i / dx_i
        xb = np.minimum(x, 1.0 - x)
        yb = np.minimum(y, 1.0 - y)
        r_bound_val = np.minimum(xb, yb)
        
        # Gradient of r_bound w.r.t. x:
        # r_bound = min(x, 1-x, y, 1-y)
        # If x is the min: grad_x = +1
        # If 1-x is the min: grad_x = -1
        # If y or 1-y is the min: grad_x = 0
        # Similarly for y.
        
        # Use argmin with tolerance for tie-breaking
        # For simplicity in soft-min context, use hard min gradient (it's a set of measure zero)
        all_bounds = np.stack([x, 1.0-x, y, 1.0-y], axis=1)  # (N, 4)
        min_idx = np.argmin(all_bounds, axis=1)  # (N,)
        
        grad_rb_dx = np.zeros(N)
        grad_rb_dy = np.zeros(N)
        for i in range(N):
            if min_idx[i] == 0:  # x is min
                grad_rb_dx[i] = 1.0
            elif min_idx[i] == 1:  # 1-x is min
                grad_rb_dx[i] = -1.0
            elif min_idx[i] == 2:  # y is min
                grad_rb_dy[i] = 1.0
            elif min_idx[i] == 3:  # 1-y is min
                grad_rb_dy[i] = -1.0

        # Set diagonal gradients
        np.fill_diagonal(d_c_dx_i, grad_rb_dx)
        np.fill_diagonal(d_c_dy_i, grad_rb_dy)
        np.fill_diagonal(d_c_dx_j, grad_rb_dx)
        np.fill_diagonal(d_c_dy_j, grad_rb_dy)

        # Now compute gradient of r_i = sum_j w_ij * c_ij
        # dr_i/dx_i = sum_j w_ij * dc_ij/dx_i = sum_j w_ij * d_c_dx_i[i,j]
        # dr_i/dy_i = sum_j w_ij * d_c_dy_i[i,j]
        # dr_i/dx_j (for j != i) = w_ij * dc_ij/dx_j = w_ij * d_c_dx_j[i,j]
        #   This contributes to dr_i/dx_j, but also dr_j/dx_i = w_ji * dc_ji/dx_i
        #   Note: c_ij = c_ji, and w_ij vs w_ji are different.
        #   The total gradient w.r.t. x_k is: sum_i dr_i/dx_k
        #   = sum_i [w_ik * dc_ik/dx_k]  (from the k-th column of row i in d_c_dx_j)
        #   + sum_i [w_ki * dc_ki/dx_k]  (from the k-th row of column i in d_c_dx_i)
        #   Wait, let me be more careful.
        
        # dr_i/dx_k for k != i: only from term j=k in the sum for r_i: w_ik * dc_ik/dx_k
        #   dc_ik/dx_k = d c_ik / dx_k. Since c_ik = 0.5*dist(i,k), dc_ik/dx_k = -0.5*dx[i,k]/dist[i,k] = d_c_dx_j[i,k]
        # dr_i/dx_i: from all j: sum_j w_ij * dc_ij/dx_i = sum_j w_ij * d_c_dx_i[i,j]
        #   But also, for j=i: w_ii * dc_ii/dx_i = w_ii * grad_rb_dx[i]
        #   For j!=i: w_ij * d_c_dx_i[i,j]
        
        # So: dr/dx_i (the i-th component of grad w.r.t. x) = 
        #   [sum_j w_ij * d_c_dx_i[i,j]]  +  [sum_{k!=i} w_ki * d_c_dx_j[k,i]]
        #   = row_sum_d_c_dx_i[i] + col_sum_w_times_d_c_dx_j[i]
        
        row_sum_dx_i = np.sum(w * d_c_dx_i, axis=1)  # (N,)
        row_sum_dy_i = np.sum(w * d_c_dy_i, axis=1)  # (N,)
        col_sum_dx_j = np.sum(w * d_c_dx_j, axis=0)  # (N,)
        col_sum_dy_j = np.sum(w * d_c_dy_j, axis=0)  # (N,)

        dr_dx = row_sum_dx_i + col_sum_dx_j  # (N,)
        dr_dy = row_sum_dy_i + col_sum_dy_j  # (N,)

        # Objective is -sum(r), so gradient is -dr/dparams
        grad = np.concatenate([-dr_dx, -dr_dy])
        
        obj_val = -np.sum(r)
        return obj_val, grad

    def check_validity(x, y, r):
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

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
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
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            # Hexagonal lattice initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            # Compute current energy
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]  # (N, N)
            dy = y[:, None] - y[None, :]  # (N, N)
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)  # Avoid division by zero on diagonal

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            # Force on i from j: -f * (x_i - x_j) / d = f * (x_j - x_i) / d
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            # Clip to valid range
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            # Evaluate new state
            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            # Acceptance criterion
            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=10):
        """Discrete best-fit local search: try moving each circle in 8 directions
        at multiple step sizes, keep the move that most improves sum of radii.
        Repeat until no improvement or max_rounds reached.
        Optimized: reduced steps and rounds for speed.
        """
        if step_sizes is None:
            step_sizes = [0.01, 0.005, 0.001]
        
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        # Normalize diagonal directions
        directions = [(dx/np.sqrt(2), dy/np.sqrt(2)) for dx, dy in directions]

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            for step in step_sizes:
                for i in range(N):
                    for dx_dir, dy_dir in directions:
                        x_new = x.copy()
                        y_new = y.copy()
                        x_new[i] += dx_dir * step
                        y_new[i] += dy_dir * step
                        
                        if x_new[i] < 0.01 or x_new[i] > 0.99:
                            continue
                        if y_new[i] < 0.01 or y_new[i] > 0.99:
                            continue
                        
                        r_new = compute_radii(x_new, y_new)
                        new_val = np.sum(r_new)
                        
                        if new_val > curr_val + 1e-10:
                            x = x_new
                            y = y_new
                            r = r_new
                            curr_val = new_val
                            improved = True
            
            if not improved:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient.
        Uses smooth soft-min objective for better gradient signal.
        Iteratively increases sharpness (beta) to converge to hard constraints.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        # Geometric progression of betas
        betas = np.geomspace(beta_start, beta_end, num_steps)
        
        current_params = params0.copy()
        best_result = None
        best_val = -np.inf
        
        def get_val_from_params(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            # Try scaling down to find a valid subset if invalid
            for scale in [0.999, 0.99, 0.95, 0.90]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf

        # Initial check
        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            best_val = init_val
            r_init = compute_radii(current_params[:N], current_params[N:])
            if check_validity(current_params[:N], current_params[N:], r_init):
                 best_result = (current_params[:N].copy(), current_params[N:].copy(), r_init, init_val)
            else:
                 for scale in [0.999, 0.99, 0.95, 0.90]:
                     r_test = r_init * scale
                     if check_validity(current_params[:N], current_params[N:], r_test):
                         best_result = (current_params[:N].copy(), current_params[N:].copy(), r_test, np.sum(r_test))
                         break

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = minimize(smooth_objective_and_grad, current_params, 
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    best_val = val
                    r_hard = compute_radii(current_params[:N], current_params[N:])
                    if check_validity(current_params[:N], current_params[N:], r_hard):
                         best_result = (current_params[:N].copy(), current_params[N:].copy(), r_hard, val)
                    else:
                         for scale in [0.999, 0.99, 0.98]:
                             r_test = r_hard * scale
                             if check_validity(current_params[:N], current_params[N:], r_test):
                                 best_result = (current_params[:N].copy(), current_params[N:].copy(), r_test, np.sum(r_test))
                                 break
            except Exception:
                pass

        # Final hard polish with very high beta
        if time.time() - start_time < DEADLINE - 2:
            beta_final = min(beta_end * 2, 2000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                current_params = res2.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    best_val = val
                    r_hard = compute_radii(current_params[:N], current_params[N:])
                    if check_validity(current_params[:N], current_params[N:], r_hard):
                         best_result = (current_params[:N].copy(), current_params[N:].copy(), r_hard, val)
                    else:
                         for scale in [0.999, 0.99, 0.98]:
                             r_test = r_hard * scale
                             if check_validity(current_params[:N], current_params[N:], r_test):
                                 best_result = (current_params[:N].copy(), current_params[N:].copy(), r_test, np.sum(r_test))
                                 break
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

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

    # Multi-start with SA + best-fit + gradient polish
    modes = ['grid', 'hex', 'random']
    num_starts = 80
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2000)

        # Best-fit local search on SA result
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish on best-fit result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try best-fit on the initial guess directly (skip SA for some starts)
        if i % 4 == 3 and time.time() - start_time < DEADLINE - 15:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if time.time() - start_time < DEADLINE - 10:
                polished2 = gradient_polish(x_bf2, y_bf2, use_smooth=True)
                if polished2 is not None:
                    xp2, yp2, rp2, vp2 = polished2
                    if vp2 > val_bf2:
                        val_bf2 = vp2
                        x_bf2, y_bf2, r_bf2 = xp2, yp2, rp2
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x = x_bf2.copy()
                best_y = y_bf2.copy()
                best_r = r_bf2.copy()

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        # First do best-fit on the best solution
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        final_polish = gradient_polish(best_x, best_y, use_smooth=True, beta_start=50.0, beta_end=1000.0, num_steps=6)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

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
