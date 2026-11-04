import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0

def construct():
    start_time = time.time()
    
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
        r = np.maximum(r, 1e-6)
        return r

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

    def make_initial_guess(seed, mode):
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
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        elif mode == 'hex':
            # Hexagonal packing initialization
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            # Generate hex points in a region larger than unit square, then clip/scale?
            # Simpler: Generate hex grid coordinates
            dx = 1.0 / 4.0
            dy = dx * np.sqrt(3) / 2.0
            for i in range(-1, 5):
                for j in range(-1, 5):
                    if idx >= N:
                        break
                    px = i * dx + (j % 2) * dx / 2.0
                    py = j * dy
                    # Center around 0.5, 0.5
                    x[idx] = px + 0.5
                    y[idx] = py + 0.5
                    idx += 1
            x += rng.uniform(-0.02, 0.02, N)
            y += rng.uniform(-0.02, 0.02, N)
        elif mode == 'ring':
            # Concentric rings
            x = np.zeros(N)
            y = np.zeros(N)
            x[0] = 0.5
            y[0] = 0.5
            idx = 1
            # Ring 1: 6 circles
            for k in range(6):
                angle = 2 * np.pi * k / 6.0
                x[idx] = 0.5 + 0.2 * np.cos(angle)
                y[idx] = 0.5 + 0.2 * np.sin(angle)
                idx += 1
            # Ring 2: 12 circles
            for k in range(12):
                angle = 2 * np.pi * k / 12.0 + np.pi/12.0
                x[idx] = 0.5 + 0.4 * np.cos(angle)
                y[idx] = 0.5 + 0.4 * np.sin(angle)
                idx += 1
            # Remaining 7 circles: scatter or fill gaps
            # For simplicity, add random points in the remaining space or just random
            while idx < N:
                x[idx] = rng.uniform(0.1, 0.9)
                y[idx] = rng.uniform(0.1, 0.9)
                idx += 1
            x += rng.uniform(-0.02, 0.02, N)
            y += rng.uniform(-0.02, 0.02, N)
        else: # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def objective_and_grad(params, beta):
        x = params[:N]
        y = params[N:2*N]
        
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        C = np.zeros((N, N + 4))
        C[:, 0] = x
        C[:, 1] = 1.0 - x
        C[:, 2] = y
        C[:, 3] = 1.0 - y
        
        half_dist = 0.5 * dist
        np.fill_diagonal(half_dist, np.inf)
        C[:, 4:] = half_dist
        
        min_C = np.min(C, axis=1, keepdims=True)
        shifted = C - min_C
        exp_terms = np.exp(-beta * shifted)
        exp_terms = np.where(np.isinf(C), 0.0, exp_terms)
        
        sum_exp = np.sum(exp_terms, axis=1)
        sum_exp = np.maximum(sum_exp, 1e-300)
        
        soft_min_radii = min_C[:, 0] - (1.0 / beta) * np.log(sum_exp)
        
        f = -np.sum(soft_min_radii)
        
        weights = exp_terms / sum_exp[:, None]
        weights = np.where(np.isinf(C), 0.0, weights)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        grad_x += weights[:, 0]
        grad_x -= weights[:, 1]
        grad_y += weights[:, 2]
        grad_y -= weights[:, 3]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_x_term = 0.5 * dx / dist
            grad_y_term = 0.5 * dy / dist
        np.fill_diagonal(grad_x_term, 0)
        np.fill_diagonal(grad_y_term, 0)
        
        weights_pairwise = weights[:, 4:]
        
        grad_x += np.sum(weights_pairwise * grad_x_term, axis=1)
        grad_y += np.sum(weights_pairwise * grad_y_term, axis=1)
        
        g = np.concatenate([-grad_x, -grad_y])
        
        return f, g

    def run_optimization_chain(x0, y0, betas, max_iter_per_stage=100):
        params = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        for beta in betas:
            if time.time() > DEADLINE - 2:
                break
            try:
                res = minimize(objective_and_grad, params, method='L-BFGS-B',
                               bounds=bounds, args=(beta,), jac=True,
                               options={'maxiter': max_iter_per_stage, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.success or res.status == 0:
                    params = res.x
                else:
                    # If it fails, keep previous params
                    pass
            except Exception:
                pass
        
        x_opt = params[:N]
        y_opt = params[N:2*N]
        r_opt = compute_radii(x_opt, y_opt)
        
        if check_validity(x_opt, y_opt, r_opt):
            return x_opt, y_opt, r_opt, np.sum(r_opt)
        else:
            # Repair
            for scale in [0.99, 0.98, 0.95, 0.9]:
                r_test = r_opt * scale
                if check_validity(x_opt, y_opt, r_test):
                    return x_opt, y_opt, r_test, np.sum(r_test)
            return None

    # Initial best
    best_x, best_y = make_initial_guess(0, 'grid')
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    modes = ['grid', 'hex', 'ring', 'random']
    num_starts = 40
    
    for i in range(num_starts):
        if time.time() > DEADLINE - 5:
            break
        
        mode = modes[i % len(modes)]
        seed = i * 13 + 7
        
        x0, y0 = make_initial_guess(seed, mode)
        
        # Annealing schedule: Smooth -> Sharp
        betas = [50.0, 200.0, 1000.0]
        
        res = run_optimization_chain(x0, y0, betas)
        
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final polish on the best solution
    if time.time() < DEADLINE - 5:
        # Try to polish the best solution with very high beta
        x0, y0 = best_x, best_y
        res = run_optimization_chain(x0, y0, [5000.0, 10000.0], max_iter_per_stage=50)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check and repair
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.85, 15):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
