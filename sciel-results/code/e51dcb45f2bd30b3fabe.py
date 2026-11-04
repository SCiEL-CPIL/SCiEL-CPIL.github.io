"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
"""
import numpy as np
from scipy.optimize import minimize
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
        # Constraints for circle i: half_dist[i, j] for j!=i, and r_bound[i]
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
        """Compute smooth objective value and analytic gradient.
        params = [x_0..x_{N-1}, y_0..y_{N-1}]
        Objective: -sum(r_i) where r_i = softmin over constraints for circle i.
        """
        x = params[:N].copy()
        y = params[N:2*N].copy()

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1e-10)  # avoid div by zero; will zero out diagonal later

        half_dist = 0.5 * dist
        # Constraints matrix: row i has [half_dist[i,0..N-1] with diagonal replaced by r_bound[i]]
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c_{ij}))
        # where c_{ij} are the constraints for circle i
        neg_beta_c = -beta * full_constraints

        # Numerically stable log-sum-exp
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        soft_min = -(max_neg + log_sum) / beta  # shape (N, 1)
        soft_min = soft_min[:, 0]  # shape (N,)

        objective_val = -np.sum(soft_min)

        # Gradient: d(-sum_i r_i)/d(params)
        # r_i = -(1/beta) * log(sum_j exp(-beta * c_{ij}))
        # d r_i / d c_{ij} = -exp(-beta * c_{ij}) / sum_k exp(-beta * c_{ik})
        #                   = -w_{ij} where w_{ij} = exp(-beta*c_{ij}) / sum_k exp(-beta*c_{ik})
        # So d r_i / d c_{ij} = -w_{ij}
        # And d(-r_i)/d c_{ij} = w_{ij}
        # Total gradient w.r.t. c_{ij}: w_{ij} (from objective -sum r_i)

        # Weights w_{ij} = exp(-beta * c_{ij}) / sum_k exp(-beta * c_{ik})
        # Using the already computed exp_terms (which is exp(neg_beta_c - max_neg)):
        # w_{ij} = exp_terms[i,j] / sum_k exp_terms[i,k]
        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)  # shape (N, N)

        # Now compute d c_{ij} / d params
        # c_{ij} = 0.5 * dist(i,j) for i != j, c_{ii} = r_bound[i]
        # dist(i,j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # d dist(i,j) / d x_i = (x_i - x_j) / dist(i,j)
        # d dist(i,j) / d x_j = (x_j - x_i) / dist(i,j) = -(x_i - x_j) / dist(i,j)
        # d c_{ij} / d x_i = 0.5 * (x_i - x_j) / dist(i,j) for i != j
        # d c_{ii} = d r_bound[i] / d x_i

        # Boundary gradient for r_bound[i]:
        # r_bound[i] = min(x_i, 1-x_i, y_i, 1-y_i)
        # For smooth approximation, use soft-min of the 4 boundary distances
        # But for gradient, let's use the subgradient of the actual min:
        # Actually, in our formulation c_{ii} = r_bound[i] = min(x_i, 1-x_i, y_i, 1-y_i)
        # This is nonsmooth. For the gradient, we'll use a smooth approximation:
        # Treat boundary as 4 additional "constraints" with soft-min.
        # But that changes the structure. Instead, let's compute gradient of the soft-min
        # including boundary contributions properly.

        # Actually, let me restructure: for each circle i, the constraints are:
        # - 0.5*dist(i,j) for all j != i
        # - x_i, 1-x_i, y_i, 1-y_i (4 boundary constraints)
        # Total N + 3 constraints per circle (N-1 neighbor + 4 boundary = N+3, but we have N columns)
        # 
        # Let me redo: build a (N, N+3) constraint matrix where:
        # columns 0..N-1: 0.5*dist(i,j) for j!=i, and for j==i we'll handle boundary separately
        # Actually this is getting complex. Let me use a different approach:
        # 
        # For the gradient, I'll compute it by considering each constraint's contribution.
        # The soft-min for circle i is over constraints {c_{i0}, c_{i1}, ..., c_{i,N-1}}
        # where c_{ij} = 0.5*dist(i,j) if i!=j, and c_{ii} = r_bound[i]
        #
        # d objective / d x_i = sum_j w_{ij} * d c_{ij} / d x_i
        #
        # For j != i: d c_{ij} / d x_i = 0.5 * (x_i - x_j) / dist(i,j)
        # For j == i: d c_{ii} / d x_i = d r_bound[i] / d x_i
        #
        # For r_bound[i] = min(x_i, 1-x_i, y_i, 1-y_i), the subgradient w.r.t. x_i is:
        # -1 if x_i is the active constraint (x_i <= 1-x_i and x_i <= y_i and x_i <= 1-y_i)
        # +1 if 1-x_i is the active constraint
        # 0 otherwise
        #
        # For a smooth approximation, I'll use a weighted average based on soft-min of the 4 boundaries.
        # But to keep it simple and correct, let me just use the hard subgradient for the diagonal.

        grad = np.zeros(2 * N)

        # For each circle i, for each constraint j:
        # Contribution to grad[x_i] from constraint j: w[i,j] * d c_{ij} / d x_i
        # Contribution to grad[y_i] from constraint j: w[i,j] * d c_{ij} / d y_i
        # Contribution to grad[x_j] from constraint i (j != i): w[i,j] * d c_{ij} / d x_j
        #   Note: d c_{ij} / d x_j = -0.5 * (x_i - x_j) / dist(i,j) = 0.5 * (x_j - x_i) / dist(i,j)

        # Vectorized computation:
        # For i != j:
        # d c_{ij} / d x_i = 0.5 * (x_i - x_j) / dist(i,j)
        # d c_{ij} / d x_j = -0.5 * (x_i - x_j) / dist(i,j)

        # Build gradient contributions from neighbor constraints
        # For each pair (i,j), i != j:
        # w[i,j] * 0.5 * (x_i - x_j) / dist(i,j) contributes to grad[x_i]
        # w[i,j] * (-0.5) * (x_i - x_j) / dist(i,j) contributes to grad[x_j]

        # Vectorized:
        # dx/dist is the unit vector component
        inv_dist = 1.0 / dist  # (N, N)
        np.fill_diagonal(inv_dist, 0.0)

        # Unit vector components
        ux = dx * inv_dist  # (x_i - x_j) / dist(i,j)
        uy = dy * inv_dist  # (y_i - y_j) / dist(i,j)

        # For circle i, constraint j (j != i):
        # d c_{ij} / d x_i = 0.5 * ux[i,j]
        # d c_{ij} / d y_i = 0.5 * uy[i,j]
        # d c_{ij} / d x_j = -0.5 * ux[i,j]
        # d c_{ij} / d y_j = -0.5 * uy[i,j]

        # grad[x_i] += sum_{j != i} w[i,j] * 0.5 * ux[i,j]
        # grad[y_i] += sum_{j != i} w[i,j] * 0.5 * uy[i,j]
        # grad[x_j] += sum_{i != j} w[i,j] * (-0.5) * ux[i,j]
        #   = -sum_{i != j} w[i,j] * 0.5 * ux[i,j]
        #   Note: summing over i for fixed j: this is the sum over rows of w[i,j]*ux[i,j]

        # Using matrix operations:
        # grad_x_from_neighbors[i] = 0.5 * sum_j w[i,j] * ux[i,j]  (for j != i, but w[i,i] and ux[i,i] are 0)
        # grad_x_from_neighbors also has contribution to other circles:
        # grad_x[i] gets 0.5 * sum_j w[i,j] * ux[i,j] from being circle i
        # grad_x[j] gets -0.5 * sum_i w[i,j] * ux[i,j] from being circle j in pair (i,j)

        # So: grad_x[i] = 0.5 * (sum_j w[i,j]*ux[i,j] - sum_j w[j,i]*ux[j,i])
        # Note: w[j,i]*ux[j,i] = w[j,i]*(-ux[i,j]) since ux[j,i] = -ux[i,j]
        # So: grad_x[i] = 0.5 * (sum_j w[i,j]*ux[i,j] + sum_j w[j,i]*ux[i,j])
        # Hmm, let me be more careful.

        # Let me just do it with explicit loops for clarity, then vectorize if needed.
        # Actually N=26, so O(N^2) is fine.

        grad_x = np.zeros(N)
        grad_y = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                wij = w[i, j]
                if wij < 1e-15:
                    continue
                # d c_{ij} / d x_i = 0.5 * ux[i,j]
                grad_x[i] += wij * 0.5 * ux[i, j]
                grad_y[i] += wij * 0.5 * uy[i, j]
                # d c_{ij} / d x_j = -0.5 * ux[i,j]
                grad_x[j] += wij * (-0.5) * ux[i, j]
                grad_y[j] += wij * (-0.5) * uy[i, j]

        # Now add boundary contributions from diagonal constraints c_{ii} = r_bound[i]
        # d r_bound[i] / d x_i: subgradient of min(x_i, 1-x_i, y_i, 1-y_i) w.r.t. x_i
        # Use smooth approximation: soft-min of the 4 boundary distances with a large beta
        # Or just use the hard subgradient (active constraint)
        # For smoothness, let's use a soft-min with beta_bound = 200
        beta_bound = 200.0
        for i in range(N):
            wii = w[i, i]  # weight for the boundary constraint
            if wii < 1e-15:
                continue
            # Soft-min gradient of min(x_i, 1-x_i, y_i, 1-y_i)
            b = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            neg_bb = -beta_bound * b
            max_bb = np.max(neg_bb)
            exp_bb = np.exp(neg_bb - max_bb)
            softmax_b = exp_bb / np.sum(exp_bb)
            # d softmin / d x_i = softmax_b[0] * d(x_i)/dx_i + softmax_b[1] * d(1-x_i)/dx_i
            #                   = softmax_b[0] * 1 + softmax_b[1] * (-1)
            #                   = softmax_b[0] - softmax_b[1]
            # d softmin / d y_i = softmax_b[2] * 1 + softmax_b[3] * (-1)
            #                   = softmax_b[2] - softmax_b[3]
            grad_x[i] += wii * (softmax_b[0] - softmax_b[1])
            grad_y[i] += wii * (softmax_b[2] - softmax_b[3])

        grad[:N] = grad_x
        grad[N:2*N] = grad_y

        return objective_val, grad

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

    def sa_optimize(x0, y0, seed, max_iters=5000):
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
            # fx[i] = sum_j f_mag[i,j] * (x_j - x_i) / d[i,j]
            # = sum_j f_mag[i,j] * (-dx[i,j]) / d[i,j]
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

        def store_best(params, val):
            nonlocal best_result, best_val
            if val > best_val:
                best_val = val
                r_hard = compute_radii(params[:N], params[N:])
                if check_validity(params[:N], params[N:], r_hard):
                    best_result = (params[:N].copy(), params[N:].copy(), r_hard, val)
                else:
                    for scale in [0.999, 0.99, 0.98, 0.95]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        # Initial check
        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            def objective_smooth(params):
                val, _ = smooth_objective_and_grad(params, beta=beta)
                return val

            def grad_smooth(params):
                _, g = smooth_objective_and_grad(params, beta=beta)
                return g

            try:
                res = minimize(objective_smooth, current_params, method='L-BFGS-B',
                               jac=grad_smooth,
                               bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Final hard polish with more iterations
        if time.time() - start_time < DEADLINE - 2:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            try:
                res2 = minimize(objective_hard, current_params, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                current_params = res2.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
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

    # Multi-start with SA
    modes = ['grid', 'hex', 'random']
    num_starts = 30
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 15:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA with more iterations (vectorized forces are faster)
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=5000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish on SA result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_sa, y_sa, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        final_polish = gradient_polish(best_x, best_y, use_smooth=True, beta_start=50.0, beta_end=2000.0, num_steps=8)
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
