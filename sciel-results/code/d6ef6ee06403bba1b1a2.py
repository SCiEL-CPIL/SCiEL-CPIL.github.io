"""
Circle packing in unit square using SA + smooth soft-min gradient polish.
Keeps the same design family: multi-start SA for global exploration,
then iterative L-BFGS-B polish with increasing soft-min sharpness.
Improvements: better SA with adaptive step, more polish iterations,
radial-sweep inspired initialization for outer ring, and a final
radius-growth polish to squeeze out extra value.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 118.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def compute_radii(x, y):
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
        r = np.maximum(r, 1e-8)
        return r

    def smooth_objective_and_grad(params, beta=50.0):
        x = params[:N]
        y = params[N:2*N]

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, 1e-10)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta_c = -beta * full_constraints
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        soft_min = -(max_neg + log_sum) / beta
        soft_min = soft_min[:, 0]

        objective_val = -np.sum(soft_min)

        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        inv_dist = 1.0 / dist
        np.fill_diagonal(inv_dist, 0.0)
        ux = dx * inv_dist
        uy = dy * inv_dist

        W_ux = w * ux
        W_T_ux = w.T * ux
        W_uy = w * uy
        W_T_uy = w.T * uy

        grad_x = 0.5 * (W_ux.sum(axis=1) + W_T_ux.sum(axis=1))
        grad_y = 0.5 * (W_uy.sum(axis=1) + W_T_uy.sum(axis=1))

        beta_bound = 500.0
        boundaries = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        neg_bb = -beta_bound * boundaries
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        softmax_b = exp_bb / np.sum(exp_bb, axis=1, keepdims=True)

        wii = w[np.arange(N), np.arange(N)]
        grad_x += wii * (softmax_b[:, 0] - softmax_b[:, 1])
        grad_y += wii * (softmax_b[:, 2] - softmax_b[:, 3])

        grad = np.concatenate([grad_x, grad_y])
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

    def grow_radii(x, y, r, max_iter=50):
        """Try to increase radii by finding the maximum uniform scale factor."""
        best_r = r.copy()
        best_val = np.sum(r)
        for _ in range(max_iter):
            if elapsed() > DEADLINE - 5:
                break
            # Try to increase each radius slightly
            improved = False
            for i in range(N):
                # Max possible radius for circle i given current positions
                r_bound_i = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                min_neighbor = np.inf
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_neighbor = min(min_neighbor, 0.5*(d - r[j]))
                max_r_i = min(r_bound_i, min_neighbor)
                if max_r_i > r[i] + 1e-10:
                    r[i] = max_r_i
                    improved = True
            if not improved:
                break
        if np.sum(r) > best_val:
            if check_validity(x, y, r):
                return r, np.sum(r)
        return best_r, best_val

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
            x += rng.uniform(-0.06, 0.06, N)
            y += rng.uniform(-0.06, 0.06, N)
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
            x += rng.uniform(-0.04, 0.04, N)
            y += rng.uniform(-0.04, 0.04, N)
        elif mode == 'ring':
            # Radial sweep: central cluster + outer ring
            # 6 central circles in hex pattern, 20 outer ring
            x = np.zeros(N)
            y = np.zeros(N)
            # Central 6: hexagonal arrangement around (0.5, 0.5)
            cx, cy = 0.5, 0.5
            rc = 0.12
            for k in range(6):
                angle = k * np.pi / 3
                x[k] = cx + rc * np.cos(angle)
                y[k] = cy + rc * np.sin(angle)
            x[0] = cx  # one in center
            y[0] = cy
            # Outer 20: ring around perimeter
            n_outer = N - 6
            for k in range(n_outer):
                angle = 2 * np.pi * k / n_outer + rng.uniform(-0.1, 0.1)
                # Place near boundary
                rr = 0.38 + rng.uniform(-0.02, 0.02)
                x[6+k] = cx + rr * np.cos(angle)
                y[6+k] = cy + rr * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=6000):
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.03
        T_end = 0.0002
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 1.5
        boundary_k = 4.0

        for it in range(max_iters):
            if elapsed() > DEADLINE - 15:
                break

            T = T_start * (cool_rate ** it)
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            fx = np.zeros(N)
            fy = np.zeros(N)

            fx += boundary_k * np.clip(0.07 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.93, 0, None)
            fy += boundary_k * np.clip(0.07 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.93, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.006)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.012 * T / T_start + 0.003
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop
            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

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

    def gradient_polish(x0, y0, beta_start=15.0, beta_end=2000.0, num_steps=8):
        params0 = np.concatenate([x0, y0])
        bounds = [(0.005, 0.995)] * (2 * N)
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
            for scale in [0.999, 0.995, 0.99, 0.98, 0.95]:
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
                    for scale in [0.999, 0.995, 0.99, 0.98, 0.95]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if elapsed() > DEADLINE - 4:
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
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Final hard polish
        if elapsed() < DEADLINE - 2:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            try:
                res2 = minimize(objective_hard, current_params, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
                current_params = res2.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

    # Initial best: grid
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
    modes = ['grid', 'hex', 'ring', 'random']
    num_starts = 35
    for i in range(num_starts):
        if elapsed() > DEADLINE - 25:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=6000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish
        if elapsed() < DEADLINE - 15:
            polished = gradient_polish(x_sa, y_sa)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish
    if elapsed() < DEADLINE - 5:
        final_polish = gradient_polish(best_x, best_y, beta_start=50.0, beta_end=5000.0, num_steps=12)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

    # Radius growth polish
    if elapsed() < DEADLINE - 2:
        grown_r, grown_val = grow_radii(best_x, best_y, best_r.copy())
        if grown_val > best_val:
            best_val = grown_val
            best_r = grown_r

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.90, 30):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
