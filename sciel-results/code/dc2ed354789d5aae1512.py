import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 295.0 # Safety margin
        d = self.dim
        rng = np.random.default_rng(12345)
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return (time.time() - start_time) < timeout

        # 1. Initial Random Sampling
        n_init = 10
        for _ in range(n_init):
            if not time_ok() or calls >= self.budget:
                break
            x = rng.uniform(-5, 5, d)
            eval_func(x)

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls += 1

        # 2. CMA-ES Implementation
        # Parameters
        popsize = max(4 + int(3 * np.log(d)), 8)
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 * mu_eff - d - 4) / ((d + 2) * (d + mu_eff))
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # State
        m = best_x.copy()
        sigma = 1.0 # Initial step size
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Helper to update C inverse sqrt
        def update_C_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            C_inv_sqrt = (eigvecs / D) @ eigvecs.T
            return C_inv_sqrt, D

        C_inv_sqrt, D = update_C_inv_sqrt(C)
        
        stagnation_counter = 0
        best_f_so_far = best_f
        
        while calls < self.budget and time_ok():
            # Generate population
            ys = []
            for i in range(popsize):
                z = rng.standard_normal(d)
                y = m + sigma * (C_inv_sqrt @ z)
                ys.append(y)
                
            # Evaluate
            fvals = [eval_func(y) for y in ys]
            
            # Sort
            order = np.argsort(fvals)
            ys_sorted = np.array(ys)[order]
            fvals_sorted = np.array(fvals)[order]
            
            # Check for improvement
            if fvals_sorted[0] < best_f_so_far:
                best_f_so_far = fvals_sorted[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Restart logic: if no improvement for 30 generations, restart
            if stagnation_counter > 30:
                # Reset CMA-ES state around current best
                m = best_x.copy()
                sigma = 1.0
                C = np.eye(d)
                C_inv_sqrt, D = update_C_inv_sqrt(C)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                stagnation_counter = 0
                continue

            # Update mean
            y_w = np.dot(weights, ys_sorted[:mu])
            
            # Update step size
            # Calculate norm in the current metric
            y_w_C_inv = C_inv_sqrt @ y_w
            norm_y_w = np.linalg.norm(y_w_C_inv)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w_C_inv
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update covariance
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - n_init) / d)) < (1.4 + 2 / (d + 1)) * chi_d else 0
            
            p_c = (1 - c_c) * p_c + c_c * h_sigma * y_w
            
            # Update C
            C = (1 + c_1 * c_c * (d + 2) - c_1) * C + c_1 * np.outer(p_c, p_c) - c_1 * np.trace(C) * np.eye(d)
            C += c_mu * np.sum([weights[i] * np.outer(ys_sorted[i] - m, ys_sorted[i] - m) for i in range(mu)], axis=0)
            
            # Update mean
            m = m + y_w
            
            # Update C_inv_sqrt
            C_inv_sqrt, D = update_C_inv_sqrt(C)

        return best_x, best_f
