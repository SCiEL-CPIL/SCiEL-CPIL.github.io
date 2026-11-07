import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def _cma_es_run(self, func, x0, sigma0, budget_run, rng):
        """
        Run CMA-ES for a specific budget.
        Returns: best_x, best_f, calls_made
        """
        n = self.dim
        # Population size: standard is 4 + floor(3 log n)
        # For n=5, 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for better exploration in low dim.
        # However, budget is small. Let's use lambda = 4 + 3*log(n)
        lam = 4 + int(3 * np.log(n))
        # Ensure lam is at least 2
        lam = max(lam, 2)
        
        # If budget is very small, adjust lam
        if budget_run < lam:
            lam = max(2, budget_run)
            
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        L = np.eye(n) # Cholesky of C
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Constants
        chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
        
        # Precompute Cholesky update helper
        # We will recompute Cholesky every step for simplicity and stability, 
        # as n=5 is very small.
        
        calls_in_run = 0
        best_x_local = x0.copy()
        best_f_local = func(x0)
        calls_in_run = 1
        
        while calls_in_run < budget_run and self._check_time():
            # Check if we can afford a full population
            if budget_run - calls_in_run < lam:
                # If not enough for full pop, do 1+1 with remaining budget? 
                # Or just break. Breaking is safer to avoid partial CMA updates.
                # But we can do single steps.
                # Let's just break to keep CMA-ES state consistent.
                break
                
            # Generate population
            # Z ~ N(0, I)
            Z = rng.standard_normal((lam, n))
            # Y = x_mean + sigma * (Z @ L)
            # Note: C = L @ L.T, so sqrt(C) = L. 
            # Standard CMA-ES uses Y = x_mean + sigma * (Z @ L) where L is Cholesky of C.
            # Actually, standard is Y = x_mean + sigma * (L.T @ Z.T).T = x_mean + sigma * (Z @ L)
            # Let's verify: Var(Y) = sigma^2 L.T I L = sigma^2 C. Correct.
            
            Y = x_mean[None, :] + sigma * (Z @ L)
            
            # Evaluate
            # Clip to box? The task says func clips. We don't need to clip before passing,
            # but it's good practice to ensure we don't pass inf/nan if L is bad.
            # C should be PD.
            f_vals = np.array([func(x) for x in Y])
            calls_in_run += lam
            
            # Sort
            idx = np.argsort(f_vals)
            Y_sorted = Y[idx]
            f_sorted = f_vals[idx]
            
            # Update global best
            if f_sorted[0] < best_f_local:
                best_f_local = f_sorted[0]
                best_x_local = Y_sorted[0].copy()
                
            # Update mean
            x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update paths
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * (norm(x_mean_new - x_mean) / sigma - chi_n)
            # h_sigma = 1 - exp(-1 / (c_sigma * (lam - 1)))
            h_sigma = 1 - np.exp(-1 / (c_sigma * (lam - 1)))
            
            # Need norm of step
            # step = x_mean_new - x_mean
            step = x_mean_new - x_mean
            norm_step = np.linalg.norm(step)
            
            # h_sigma check
            # h_sigma = 1 if norm_step / sigma < (1 - exp(-1/(c_sigma*(lam-1)))) * chi_n
            # Actually, standard formula:
            # h_sigma = 1 - exp(-1 / (c_sigma * (lam - 1)))
            # if norm_step / sigma < h_sigma * chi_n: h_sigma_flag = 1 else 0
            # Wait, the condition is usually:
            # h_sigma = 1 - exp(-1 / (c_sigma * (lam - 1)))
            # h_sigma_flag = 1 if norm_step / sigma < h_sigma * chi_n else 0
            
            # Let's use the simpler threshold:
            # h_sigma_flag = 1 if norm_step < sigma * chi_n * (1 - exp(-1/(c_sigma*(lam-1)))) else 0
            # Actually, Hansen's implementation:
            # h_sigma = 1 - exp(-1 / (c_sigma * (lam - 1)))
            # if norm(step) / sigma < h_sigma * chi_n: h_sigma_flag = 1 else 0
            
            h_sigma_flag = 1 if norm_step / sigma < h_sigma * chi_n else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma_flag * (step / sigma - chi_n * L.T @ (np.linalg.inv(L) @ np.ones(n)) ) 
            # Wait, the term is (x_mean_new - x_mean) / sigma - chi_n * e
            # where e is a unit vector? No.
            # Standard: p_sigma = (1-c_sigma)p_sigma + c_sigma * h_sigma * ( (x_mean_new - x_mean) / sigma - chi_n * 1 )
            # The vector 1 is not used. It's just the norm correction.
            # Actually, the update is:
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * ( (x_mean_new - x_mean) / sigma )
            # And the chi_n term is used in the h_sigma condition, not in the update itself?
            # Let's check Hansen's code.
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma*h_sigma*(norm(x_mean_new-x_mean)/sigma - chi_n) * (x_mean_new-x_mean)/norm(...)
            # No, that's for direction.
            # Correct update:
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * ( (x_mean_new - x_mean) / sigma )
            # Wait, the chi_n is subtracted from the norm in the h_sigma calculation.
            # Let's stick to the most common simplified version:
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma_flag * (step / sigma)
            
            # h_c
            h_c = 1 - np.exp(-(1 + c_sigma) / (c_c * (lam - 1)))
            # Condition for h_c
            # if norm(p_sigma) / sqrt(1 - (1-c_sigma)^{2*iter}) < (1.4 + 2/(n+1)) * chi_n: h_c_flag = 1 else 0
            # We don't track iter easily here, but for small budgets it's fine to approximate or use a fixed threshold.
            # Let's use a simple check:
            h_c_flag = 1 if np.linalg.norm(p_sigma) < (1.4 + 2.0 / (n + 1)) * chi_n * np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_run // lam))) else 0
            
            p_c = (1 - c_c) * p_c + c_c * h_c_flag * (step / sigma)
            
            # Update Covariance
            # Rank-one update
            y = step / sigma
            C = (1 + c_1 * h_c_flag * h_c_flag - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Y_diff = Y_sorted[:mu] - x_mean
            # Y_diff_norm = Y_diff / sigma
            # C += c_mu * sum( w_i * outer(Y_diff_norm_i, Y_diff_norm_i) )
            Y_diff = Y_sorted[:mu] - x_mean
            Y_diff_norm = Y_diff / sigma
            rank_mu_term = np.zeros((n, n))
            for i in range(mu):
                rank_mu_term += weights[i] * np.outer(Y_diff_norm[i], Y_diff_norm[i])
            
            C += c_mu * rank_mu_term
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Sigma
            # sigma = sigma * exp( c_sigma / d_sigma * (norm(p_sigma)/chi_n - 1) )
            # d_sigma approximation: 1 + 2*max(0, sqrt((mu_eff-1)/(n+1)) - 1) + c_sigma
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update Mean
            x_mean = x_mean_new
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity and sigma
                C = np.eye(n)
                L = np.eye(n)
                sigma *= 0.5
                p_c = np.zeros(n)
                p_sigma = np.zeros(n)
                
        return best_x_local, best_f_local, calls_in_run

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        
        # Multi-start strategy
        # Start with a few diverse initial points
        # For 5D, 3-5 restarts is good.
        # Budget 500. 
        # If we use 5 restarts, each gets 100 budget.
        # Or adaptive: give more budget to better runs? 
        # Simple: Fixed number of restarts, equal budget.
        
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        # Ensure at least some budget
        budget_per_restart = max(budget_per_restart, 10)
        
        # Initial points
        # Use a spread of points
        # 1. Origin
        # 2. Random
        # 3. Random
        # 4. Random
        # 5. Random
        
        initial_points = [np.zeros(self.dim)]
        for _ in range(n_restarts - 1):
            initial_points.append(rng.uniform(-5, 5, self.dim))
        
        global_best_x = None
        global_best_f = np.inf
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if not self._check_time():
                break
                
            x0 = initial_points[i]
            sigma0 = 2.0 if i == 0 else 1.0 # Larger step for first, smaller for others? 
            # Actually, random restarts benefit from larger sigma to explore.
            # Let's use sigma0 = 2.0 for all.
            sigma0 = 2.0
            
            # Check if we have enough budget for this run
            remaining = self.budget - self.calls
            if remaining < 10:
                # If very little budget left, just do 1+1 or random search?
                # CMA-ES needs at least lam calls.
                # If remaining < lam, just do random local search.
                pass
                
            x_best_run, f_best_run, calls_made = self._cma_es_run(func, x0, sigma0, min(remaining, budget_per_restart), rng)
            
            self.calls += calls_made
            
            if f_best_run < global_best_f:
                global_best_f = f_best_run
                global_best_x = x_best_run.copy()
                
            # If we found a very good solution early, we might stop?
            # No, AOCC rewards progress. We should use all budget.
            
        # If we still have budget left (e.g. due to break conditions in CMA-ES)
        # and time is left, do a final 1+1 polish on the best point found
        if self.calls < self.budget and self._check_time() and global_best_x is not None:
            x = global_best_x.copy()
            f = global_best_f
            sigma = 0.5
            while self.calls < self.budget and self._check_time():
                z = rng.standard_normal(self.dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                self.calls += 1
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break
            
            if f < global_best_f:
                global_best_f = f
                global_best_x = x
        
        self.best_x = global_best_x
        self.best_f = global_best_f
        return self.best_x, self.best_f
