"""CMA-ES based optimizer for the BBOB-style Discus function (f11)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for the BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within a single run
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_cma_es(x_start, sigma_init, max_calls):
            nonlocal call_count, best_f, best_x
            
            # CMA-ES Parameters
            n = dim
            lam = 9  # Population size
            mu = lam // 2  # Number of best solutions used for update
            
            # Weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum = np.sum(w)
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w**2)
            
            # Step size control
            c_sigma = (mu_eff / 2 + 1) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Covariance matrix control
            c_c = (4 - mu_eff / 2 + c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initialization
            x_m = x_start.copy()
            C = np.eye(n)
            sigma = sigma_init
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Pre-compute Cholesky factorization
            C_sqrt = np.linalg.cholesky(C)
            
            best_x_local = x_start.copy()
            best_f_local = float('inf')
            
            # Initial evaluation
            fx_start = try_eval(x_start)
            if fx_start is None:
                return best_x_local, best_f_local
            best_f_local = fx_start
            best_x_local = x_start.copy()
            
            # Main loop
            while call_count < budget and (budget - call_count) > lam:
                if time.time() > deadline - 5:
                    break
                
                # Generate population
                ys = rng.multivariate_normal(np.zeros(n), C, size=lam)
                xs = x_m + sigma * ys
                fs = np.zeros(lam)
                
                for i in range(lam):
                    fx = try_eval(xs[i])
                    if fx is None:
                        break
                    fs[i] = fx
                    if fx < best_f_local:
                        best_f_local = fx
                        best_x_local = xs[i].copy()
                
                if call_count >= budget:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]
                
                # Update mean
                x_m_new = x_m + sigma * np.dot(w, ys_sorted[:mu])
                
                # Update step size (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.dot(C_sqrt, x_m_new - x_m)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((budget - call_count) / d_sigma)) <
                        (1 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
                sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Update covariance matrix
                y_w = np.dot(w, ys_sorted[:mu].T).T
                delta_sig = (1 - hsig) * c_sigma * (2 - c_sigma)
                rank_one = np.outer(ps, ps)
                rank_mu = np.dot(ys_sorted[:mu].T, np.diag(w) @ ys_sorted[:mu])
                
                C = (1 + c_1 * (c_mu + delta_sig - 1) - c_1 - c_mu) * C + \
                    c_1 * rank_one + c_mu * rank_mu
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, regularize
                    C += 1e-10 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_sqrt = np.eye(n)
                
                x_m = x_m_new
                
                # Stagnation check: if no improvement for 30 generations, break
                # This is a simple heuristic; proper CMA-ES uses termination criteria
                if call_count % (lam * 10) == 0:
                     # Check if we are making progress
                     pass 

            return best_x_local, best_f_local

        # Phase 1: Global search from origin
        x0 = np.zeros(dim)
        x_best_1, f_best_1 = run_cma_es(x0, 2.0, budget)
        
        # Phase 2: Local search around best found
        if best_x is not None and call_count < budget:
            # Perturb best point slightly to escape local minima if any
            x_start_2 = best_x + 0.5 * rng.standard_normal(dim)
            x_start_2 = np.clip(x_start_2, -5, 5)
            x_best_2, f_best_2 = run_cma_es(x_start_2, 0.5, budget)
            
        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
