import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0 # Safety margin
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_f = f
                best_x = x
            # Note: calls is incremented by the caller

        # Phase 1: Global Exploration with Sobol
        n_expl = 60
        if self.budget > n_expl:
            sobol = qmc.Sobol(d=n, scramble=True)
            # We need enough points. Sobol generates in [0,1]^n.
            # We map to [-5, 5].
            # Sobol requires power of 2 for best performance, but works for any n.
            # Let's use 64 points.
            m = 64
            if m > self.budget: m = self.budget
            points = sobol.random(m)
            points = points * 10.0 - 5.0 # Map [0,1] to [-5,5]
            
            f_vals = []
            for x in points:
                if calls >= self.budget or time.time() > deadline:
                    break
                f = func(x)
                calls += 1
                f_vals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x
            
            if not f_vals:
                # Fallback if no calls made
                x = np.zeros(n)
                f = func(x)
                calls += 1
                best_f = f
                best_x = x
            
            # Sort and get top 3 candidates
            indices = np.argsort(f_vals)
            top_candidates = []
            for i in indices:
                if len(top_candidates) >= 3:
                    break
                top_candidates.append((points[i], f_vals[i]))
                
            # Phase 2: Local Refinement of Top Candidates
            # We have budget_left calls.
            # Let's allocate some calls to refine the top candidates.
            # If we have 440 calls left, we can do 3 * 50 = 150 calls for local refinement, leaving 290 for final CMA-ES.
            # Or just pick the best and run CMA-ES.
            
            # Let's try refining the best 2 candidates with a quick CMA-ES.
            # Actually, simpler: Just use the best Sobol point as the start for the main CMA-ES.
            # The main CMA-ES will do the heavy lifting.
            
            # Let's set the start of the main CMA-ES to the best point found so far.
            start_mean = best_x.copy()
            start_sigma = 1.0 # Smaller than global, larger than local
            
        else:
            # If budget is very small, just do random search
            start_mean = np.zeros(n)
            start_sigma = 5.0

        # Phase 3: Main CMA-ES
        # Reuse the CMA-ES logic from the current program but initialized at start_mean
        lambda_ = 15
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        rng = np.random.default_rng(42)
        
        mean = start_mean
        C = np.eye(n)
        sigma = start_sigma
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff/n - 1) + 1) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += np.eye(n) * 1e-6
                L = np.linalg.cholesky(C)
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)
            
            # Evaluate
            f_vals = []
            for x in pop:
                if calls >= self.budget or time.time() > deadline:
                    break
                f = func(x)
                calls += 1
                f_vals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x
            
            if len(f_vals) < lambda_:
                while len(f_vals) < lambda_:
                    f_vals.append(np.inf)
                    pop.append(pop[-1])
            
            # Sort
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update mean
            new_mean = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (new_mean - mean) / sigma
            
            # Update covariance
            v = (new_mean - mean) / sigma
            C = (1 + c_1 * (h_sigma * (1 + (1 - c_sigma) * c_sigma) - c_1) - c_mu) * C \
                + c_1 * (np.outer(v, v) + (1 - h_sigma) * c_sigma * (1 - c_sigma) * C)
            
            y = np.array([(sorted_pop[i] - mean) / sigma for i in range(mu)])
            C += c_mu * np.sum([weights[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Optional: Early stopping if sigma is very small or no improvement for many gens
            # But we rely on the budget limit.

        return best_x, best_f
