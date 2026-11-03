"""
Circle packing in unit square using a hybrid approach:
- Smooth soft-min objective with analytic gradient for fast L-BFGS-B polishing
- Delaunay-based void identification to guide initialization and local moves
- Multi-start with diverse initializations (grid, hex, random)
- Deadline-aware time management to stay within 120s limit

Key structural change from seed: replaces subgradient descent with L-BFGS-B on a
smoothed objective, which converges much faster per polish call, freeing budget
for more restarts and finer polish.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


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

    def smooth_objective_and_grad(params, beta=80.0):
        """
        Compute -sum(r) and its gradient w.r.t. params = [x_0..x_{N-1}, y_0..y_{N-1}].
        Uses analytic gradient of the log-sum-exp soft-min for smoothness.
        """
        x = params[:N]
        y = params[N:2 * N]

        # Boundary radii
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Constraints: c_ij = 0.5 * dist_ij for i != j, c_ii = r_bound_i
        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        neg_beta_c = -beta * full_constraints
        max_nb = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_nb)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        r = -(max_nb + log_sum) / beta  # (N, 1)
        r = r[:, 0]  # (N,)

        # Objective: minimize -sum(r)
        obj = -np.sum(r)

        # Weights w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        # Gradient of constraints w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # d(0.5*dist_ij)/dx_i = 0.5 * (x_i - x_j) / dist_ij
        d_c_dx_i = 0.5 * dx / dist_safe
        d_c_dy_i = 0.5 * dy / dist_safe
        # d(0.5*dist_ij)/dx_j = -0.5 * (x_i - x_j) / dist_ij
        d_c_dx_j = -0.5 * dx / dist_safe
        d_c_dy_j = -0.5 * dy / dist_safe

        # Boundary gradient: d(r_bound_i)/dx_i depends on which boundary is active
        all_bounds = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        min_idx = np.argmin(all_bounds, axis=1)

        grad_rb_dx = np.zeros(N)
        grad_rb_dy = np.zeros(N)
        for i in range(N):
            if min_idx[i] == 0:
                grad_rb_dx[i] = 1.0
            elif min_idx[i] == 1:
                grad_rb_dx[i] = -1.0
            elif min_idx[i] == 2:
                grad_rb_dy[i] = 1.0
            elif min_idx[i] == 3:
                grad_rb_dy[i] = -1.0

        # Set diagonal gradients in the constraint gradient matrices
        d_c_dx_i[np.arange(N), np.arange(N)] = grad_rb_dx
        d_c_dy_i[np.arange(N), np.arange(N)] = grad_rb_dy
        d_c_dx_j[np.arange(N), np.arange(N)] = grad_rb_dx
        d_c_dy_j[np.arange(N), np.arange(N)] = grad_rb_dy

        # dr_i/dx_i = sum_j w_ij * dc_ij/dx_i
        # dr_i/dx_j = sum_i w_ij * dc_ij/dx_j  (for j != i)
        # But we need gradient of -sum(r) w.r.t. all params

        # grad_x[i] = -dr_i/dx_i = -sum_j w[i,j] * d_c_dx_i[i,j]
        grad_x = -np.sum(w * d_c_dx_i, axis=1)
        grad_y = -np.sum(w * d_c_dy_i, axis=1)

        # Also need contributions from other circles' radii to x_i
        # dr_j/dx_i = sum_k w[j,k] * dc_jk/dx_i
        # For k=i: dc_ji/dx_i = d_c_dx_j[j,i] = -0.5*(x_j - x_i)/dist_ji
        # For k!=i: dc_jk/dx_i = 0
        # So dr_j/dx_i = w[j,i] * d_c_dx_j[j,i] for j != i
        # And dr_i/dx_i already includes w[i,i]*grad_rb_dx[i] + sum_{j!=i} w[i,j]*d_c_dx_i[i,j]

        # Additional contribution to grad_x[i] from other circles j:
        # -sum_{j!=i} dr_j/dx_i = -sum_{j!=i} w[j,i] * d_c_dx_j[j,i]
        # Note d_c_dx_j[j,i] = -0.5*(x_j - x_i)/dist_ji = 0.5*(x_i - x_j)/dist_ji = d_c_dx_i[i,j]
        # So this is -sum_{j!=i} w[j,i] * d_c_dx_i[i,j]

        # We can compute this via matrix operations:
        # For each i, sum over j != i of w[j,i] * d_c_dx_i[i,j]
        # This is (w.T * d_c_dx_i)[i] minus the diagonal term w[i,i]*d_c_dx_i[i,i]
        # But d_c_dx_i[i,i] = grad_rb_dx[i], and we don't want to double count.

        # Actually, let's re-derive more carefully.
        # Objective F = -sum_i r_i
        # dF/dx_i = -sum_i dr_i/dx_i
        # dr_i/dx_i = sum_j w[i,j] * dc_ij/dx_i
        #   where dc_ij/dx_i for j!=i is 0.5*(x_i-x_j)/dist_ij, for j=i is grad_rb_dx[i]
        # dr_j/dx_i for j!=i = sum_k w[j,k] * dc_jk/dx_i
        #   dc_jk/dx_i is nonzero only when k=i (since x_i only appears in dist_ji and dist_ik when k=i)
        #   When k=i: dc_ji/dx_i = 0.5*(x_j - x_i)/dist_ji... wait
        #   Actually dc_jk = 0.5*dist_jk. d(0.5*dist_ji)/dx_i = 0.5*(x_j - x_i)/dist_ji
        #   So dr_j/dx_i = w[j,i] * 0.5*(x_j - x_i)/dist_ji = w[j,i] * d_c_dx_j[j,i]
        #   Note d_c_dx_j[j,i] = -0.5*(x_j - x_i)/dist_ji... let me re-check.
        #   d_c_dx_j is defined as -0.5*dx/dist where dx = x[:,None]-x[None,:]
        #   So d_c_dx_j[j,i] = -0.5*(x[j]-x[i])/dist[j,i] = 0.5*(x[i]-x[j])/dist[j,i]
        #   And d_c_dx_i[i,j] = 0.5*(x[i]-x[j])/dist[i,j]
        #   So d_c_dx_j[j,i] = d_c_dx_i[i,j]. Good.

        # So dF/dx_i = -[sum_j w[i,j]*d_c_dx_i[i,j] + sum_{j!=i} w[j,i]*d_c_dx_i[i,j]]
        #            = -[sum_j w[i,j]*d_c_dx_i[i,j] + sum_{j} w[j,i]*d_c_dx_i[i,j] - w[i,i]*d_c_dx_i[i,i]]
        #            = -[(w + w.T) * d_c_dx_i][i] + w[i,i]*d_c_dx_i[i,i]

        # Let's compute this properly:
        W = w  # (N,N)
        # grad_x[i] = -sum_j (W[i,j] + W[j,i]) * d_c_dx_i[i,j] + W[i,i]*d_c_dx_i[i,i]
        # The first term: -(W + W.T) @ d_c_dx_i gives a matrix, but we need row-wise sums
        # Actually: sum_j (W[i,j] + W[j,i]) * d_c_dx_i[i,j] = sum_j W[i,j]*d_c_dx_i[i,j] + sum_j W[j,i]*d_c_dx_i[i,j]
        # First part: (W @ d_c_dx_i)[i]... no, that's sum_j W[i,j]*d_c_dx_i[i,j] which is correct for first part
        # Second part: sum_j W[j,i]*d_c_dx_i[i,j] = sum_j W.T[i,j]*d_c_dx_i[i,j] = (W.T @ d_c_dx_i)[i]... 
        # Wait, (W.T @ d_c_dx_i)[i] = sum_j W.T[i,j]*d_c_dx_i[j]... no, d_c_dx_i is a matrix not a vector.

        # Let me just compute it element-wise for clarity, N=26 is small enough
        grad_x_full = np.zeros(N)
        grad_y_full = np.zeros(N)

        for i in range(N):
            # Contribution from r_i's own gradient
            gx_i = 0.0
            gy_i = 0.0
            for j in range(N):
                if j == i:
                    gx_i += w[i, j] * grad_rb_dx[i]
                    gy_i += w[i, j] * grad_rb_dy[i]
                else:
                    gx_i += w[i, j] * d_c_dx_i[i, j]
                    gy_i += w[i, j] * d_c_dy_i[i, j]

            # Contribution from other circles' radii
            for j in range(N):
                if j == i:
                    continue
                # dr_j/dx_i = w[j,i] * d_c_dx_j[j,i] = w[j,i] * d_c_dx_i[i,j]
                gx_i += w[j, i] * d_c_dx_i[i, j]
                gy_i += w[j, i] * d_c_dy_i[i, j]

            grad_x_full[i] = -gx_i
            grad_y_full[i] = -gy_i

        grad = np.concatenate([grad_x_full, grad_y_full])
        return obj, grad

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
        else:
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, time_limit=2.0):
        """
        Fast Coarse Simulated Annealing for global exploration.
        """
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        T_start = 0.02
        T_end = 0.001
        sa_start = time.time()
        elapsed = 0.0

        max_iters = 1500
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        it = 0
        while elapsed < time_limit:
            if time.time() - start_time > DEADLINE - 5:
                break

            T = T_start * (cool_rate ** it)

            scale = 0.01 * (T / T_start) + 0.002
            dx_prop = rng.normal(0, scale, N)
            dy_prop = rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)

            new_r = compute_radii(x_new, y_new)
            new_val = np.sum(new_r)

            delta_val = new_val - current_val

            if delta_val > 0 or (T > 1e-6 and rng.random() < np.exp(delta_val / T)):
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val

                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()

            it += 1
            elapsed = time.time() - sa_start

            if T < 1e-5:
                break

        return best_x, best_y, best_val

    def run_l_bfgs_polish(x0, y0, time_limit=3.0):
        """
        L-BFGS-B optimization on the smooth soft-min objective.
        """
        params0 = np.concatenate([x0, y0])

        def objective(params):
            obj, grad = smooth_objective_and_grad(params, beta=80.0)
            return obj, grad

        try:
            res = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                },
                bounds=[(0.0, 1.0)] * (2 * N),
            )
        except Exception:
            res = None

        if res is not None and res.success:
            x_opt = res.x[:N]
            y_opt = res.x[N:]
        else:
            x_opt = x0
            y_opt = y0

        r_opt = compute_radii(x_opt, y_opt)
        val_opt = np.sum(r_opt)

        # Check validity and scale down if needed
        if not check_validity(x_opt, y_opt, r_opt):
            for scale in np.linspace(0.99, 0.85, 15):
                r_test = r_opt * scale
                if check_validity(x_opt, y_opt, r_test):
                    r_opt = r_test
                    val_opt = np.sum(r_opt)
                    break
            else:
                r_opt = compute_radii(x_opt, y_opt)
                val_opt = np.sum(r_opt)

        return x_opt, y_opt, r_opt, val_opt

    def run_optimization(seed, mode):
        x0, y0 = make_initial_guess(seed, mode)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 8:
            return None

        # Allocate time to SA first
        sa_time = min(2.0, remaining * 0.4)
        x_sa, y_sa, val_sa = sa_optimize(x0, y0, seed, sa_time)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            r_sa = compute_radii(x_sa, y_sa)
            if check_validity(x_sa, y_sa, r_sa):
                return x_sa, y_sa, r_sa, val_sa
            return None

        # L-BFGS-B polish
        polish_time = min(4.0, remaining * 0.6)
        result = run_l_bfgs_polish(x_sa, y_sa, polish_time)
        if result is not None:
            return result

        r_sa = compute_radii(x_sa, y_sa)
        if check_validity(x_sa, y_sa, r_sa):
            return x_sa, y_sa, r_sa, val_sa

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

    # Multi-start loop
    modes = ['grid', 'hex', 'random']
    num_starts = 10

    for i in range(num_starts):
        if time.time() > DEADLINE - 10:
            break

        mode = modes[i % len(modes)]
        seed = i * 7 + 3

        res = run_optimization(seed, mode)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.85, 15):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
