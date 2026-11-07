# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Global exploration with Latin Hypercube + random ---
        n_global = 40
        candidates = []
        
        # Latin Hypercube sampling for better coverage
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            if i < d * 5:  # First d*5 points: LHS
                # Simple LHS: for each dimension, divide into 5 equal intervals
                # Assign each point to a unique cell
                x = np.zeros(d)
                for j in range(d):
                    interval = (i // d) % 5
                    frac = (i % d) / d
                    x[j] = -5 + (interval + frac) * 2.0
            else:
                x = rng.uniform(-5, 5, d)
            
            f = func(x)
            used_calls += 1
            candidates.append((f, x.copy()))
            try_update(x, f)

        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # --- Phase 2: CMA-ES implementation ---
        def cma_es(x0, f0, max_calls, sigma0=0.5):
            nonlocal used_calls, best_f, best_x
            
            # CMA-ES parameters
            lambda_ = 4 * d  # population size
            mu = lambda_ // 2  # recombination weight
            # Compute weights
            w = np.zeros(mu)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            
            # Initial state
            x_m = x0.copy()  # mean
            sigma = sigma0
            C = np.eye(d)  # covariance
            C_sqrt_inv = np.eye(d)  # inverse square root of C
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            h_sigma = (1 + 1. / (2 * c_sigma)) * np.sqrt((mu_eff + 2) / (d + 2))
            
            # Precompute
            D, V = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            C_sqrt = V @ np.diag(D) @ V.T
            C_sqrt_inv = V @ np.diag(1.0 / D) @ V.T
            
            # Initial population
            pop = []
            pop_f = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                x_i = x_m + sigma * (C_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                if used_calls >= max_calls or check_time():
                    return
                f_i = func(x_i)
                used_calls += 1
                pop.append(x_i)
                pop_f.append(f_i)
                try_update(x_i, f_i)
            
            while used_calls < max_calls and not check_time():
                # Sort population
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]
                
                # Update mean
                x_m_old = x_m.copy()
                x_m = sum(w[i] * pop[i] for i in range(mu))
                
                # Update step-length control
                x_m_diff = x_m - x_m_old
                if sigma > 0:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * C_sqrt_inv @ (x_m_diff / sigma)
                
                # Update evolution path
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (used_calls - lambda_))) < (1.4 + 2 / (d + 1)) * h_sigma:
                    h_sigma = 1.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / chi_n * C_sqrt_inv @ (x_m_diff / sigma)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank1 = np.outer(p_c, p_c)
                rankmu = np.zeros((d, d))
                for i in range(mu):
                    z_i = C_sqrt_inv @ (pop[i] - x_m_old) / sigma
                    rankmu += w[i] * np.outer(z_i, z_i)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                D, V = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C = V @ np.diag(D) @ V.T
                C_sqrt = V @ np.diag(np.sqrt(D)) @ V.T
                C_sqrt_inv = V @ np.diag(1.0 / np.sqrt(D)) @ V.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-10, 10.0))
                
                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(lambda_):
                    if used_calls >= max_calls or check_time():
                        return
                    z = rng.standard_normal(d)
                    x_i = x_m + sigma * (C_sqrt @ z)
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    used_calls += 1
                    new_pop.append(x_i)
                    new_pop_f.append(f_i)
                    try_update(x_i, f_i)
                
                pop = new_pop
                pop_f = new_pop_f
                
                # Check convergence
                if np.std(pop_f[:mu]) < 1e-8 and np.linalg.norm(sigma * C_sqrt) < 1e-6:
                    break
            
            # Update global best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

        # --- Phase 3: Run CMA-ES from top candidates ---
        # Select top 3 distinct starting points
        starts = []
        for f, x in candidates:
            is_distinct = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append((f, x))
            if len(starts) >= 3:
                break
        
        if not starts:
            starts = [(candidates[0][0], candidates[0][1])]
        
        remaining_calls = budget - used_calls
        calls_per_cma = max(50, remaining_calls // len(starts))
        
        for i, (f_start, x_start) in enumerate(starts):
            if check_time() or used_calls >= budget:
                break
            allocated = min(calls_per_cma, budget - used_calls - 20)
            if allocated < 20:
                allocated = max(10, budget - used_calls)
            cma_es(x_start, f_start, used_calls + allocated, sigma0=1.0)
        
        # --- Phase 4: Final polish with (1+1)-ES ---
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(100):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                sigma = float(np.clip(sigma, 1e-8, 5.0))
                if sigma < 1e-7:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
