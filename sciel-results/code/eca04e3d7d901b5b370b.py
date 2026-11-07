import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse exploration to find promising basins
        # For Rastrigin, local minima are near integer coordinates (after inverse rotation)
        # Sample diverse points including near-integer points and random points
        n_init = 40
        candidates = []

        # Try origin and near-integer points
        eval_func(np.zeros(dim))
        candidates.append((best_f, np.zeros(dim)))

        # Sample near-integer points (Rastrigin local minima are near integers)
        int_points = []
        for coord in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            if len(int_points) < 15:
                int_points.append(np.array([coord] * dim))
        # Add some random near-integer points
        for _ in range(10):
            pt = rng.integers(-4, 5, dim).astype(float)
            pt += rng.uniform(-0.5, 0.5, dim)
            int_points.append(pt)

        for pt in int_points[:20]:
            if calls >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = eval_func(pt)
            candidates.append((f, pt))

        # Random exploration
        while len(candidates) < n_init:
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start CMA-ES
        # Run CMA-ES from the best initial points with restarts on stagnation
        
        def cma_es(start_x, max_evals, sigma_init=1.0):
            """Simple CMA-ES implementation"""
            nonlocal calls, best_x, best_f, deadline
            
            n = dim
            pop_size = 4 + int(3 * np.log(n))  # ~12 for n=5
            pop_size = min(pop_size, max(6, max_evals // 10))
            
            x_mean = start_x.copy()
            sigma = sigma_init
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            
            # Evolution path
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # Update rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((n + 2) ** 2 + mu_eff))
            
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Initial evaluation
            f_mean = eval_func(x_mean)
            best_local_x = x_mean.copy()
            best_local_f = f_mean
            
            max_iter = max_evals // pop_size + 1
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate population
                ys = []
                xs = []
                fs = []
                
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = rng.standard_normal(n)
                    x_new = x_mean + sigma * (C_sqrt @ y)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = eval_func(x_new)
                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_new)
                
                if len(xs) < 2:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update best
                if fs[0] < best_local_f:
                    best_local_f = fs[0]
                    best_local_x = xs[0].copy()
                
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                
                # Update mean
                y_w = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
                x_mean_new = x_mean + sigma * (C_sqrt @ y_w)
                
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt_inv @ y_w)
                delta_sig = np.abs(np.linalg.norm(ps) / chi_n - 1)
                pc = (1 - c_c) * pc + c_c * delta_sig * (x_mean_new - x_mean) / (sigma * np.sqrt(1 - (1 - c_c) ** 2)) if sigma > 1e-10 else (1 - c_c) * pc
                
                # Update covariance
                rank_1 = np.outer(pc, pc)
                y_w_normalized = y_w / (sigma + 1e-10)
                rank_mu = np.sum([weights[i] * np.outer(ys[i], ys[i]) for i in range(mu)], axis=0)
                C = (1 + c_1 * delta_sig * (n + 2) - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
                C = (C + C.T) / 2  # Symmetrize
                
                # Update C_sqrt (approximate via eigendecomposition)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                x_mean = x_mean_new
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)
                
                # Stagnation check: if no improvement in last 2*pop_size evaluations, break
                if it > 0 and it % pop_size == 0:
                    # Check if we've made progress in the last pop_size iterations
                    # Simple check: if sigma is very small or mean hasn't moved much
                    if sigma < 0.01:
                        break
            
            return best_local_x, best_local_f

        # Run CMA-ES from top candidates
        top_k = min(5, len(candidates))
        cma_evals_per_run = 80
        
        for k in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            start_x = candidates[k][1].copy()
            sigma_init = 1.5 if k == 0 else 0.8
            
            bx, bf = cma_es(start_x, cma_evals_per_run, sigma_init)
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If we found a good solution, reduce remaining evals
            if best_f < -50:
                cma_evals_per_run = 40
        
        # Phase 3: Final fine-tuning with small-step CMA-ES or (1+1)-ES
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 20:
                # Fine-tune around best_x
                fine_x = best_x.copy()
                fine_f = best_f
                sigma = 0.05
                
                for _ in range(min(remaining, 60)):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = fine_x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    f_y = eval_func(y)
                    if f_y < fine_f:
                        fine_x = y
                        fine_f = f_y
                        sigma *= 0.95
                    else:
                        sigma *= 0.9
                    sigma = max(sigma, 0.001)
                
                if fine_f < best_f:
                    best_f = fine_f
                    best_x = fine_x.copy()

        return best_x, best_f
