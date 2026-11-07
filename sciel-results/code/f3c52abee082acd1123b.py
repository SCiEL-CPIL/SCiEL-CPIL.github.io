import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0
        self.start_time = 0
        self.deadline = 0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # 20s safety margin before 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        dim = self.dim
        budget = self.budget

        # Wrap function to count calls, check time, and track best
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time Limit Exceeded")
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                try:
                    x = x.reshape(dim)
                except:
                    x = np.zeros(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def run_cma_es(x0, sigma0, max_calls, lam=None):
            """
            Robust CMA-ES implementation.
            Returns best_x, best_f found within max_calls.
            """
            if max_calls <= 0:
                return x0.copy(), float('inf')
            
            # Evaluate initial point
            f0 = wrapped_func(x0)
            local_best_x = x0.copy()
            local_best_f = f0
            local_calls = 1
            
            if local_calls >= max_calls:
                return local_best_x, local_best_f

            # Parameters
            if lam is None:
                lam = max(4, int(4 + 3 * math.log(dim)))  # e.g., dim=5 -> lam=9
            mu = max(1, lam // 2)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
            c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
            c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1.0 - c_1, 2.0 * ((mu_eff + 2.0) / (dim + 2.0) - 1.0))
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute C^{-1/2} initially (identity)
            C_inv_sqrt = np.eye(dim)
            
            chi_n = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))
            
            stagnation_counter = 0
            best_improvement = 1e-10
            
            while local_calls < max_calls:
                if time.time() > self.deadline:
                    break
                
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break
                
                # Sample offspring
                # Use Cholesky for sampling if possible, else eigen
                try:
                    L = np.linalg.cholesky(C)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * (L @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                except np.linalg.LinAlgError:
                    # Fallback to eigen if C is not PD
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    std_eig = np.sqrt(eigvals)
                    samples = []
                    for _ in range(n_off):
                        z = np.random.randn(dim)
                        y = x_mean + sigma * ((eigvecs * std_eig) @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = wrapped_func(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                
                if time.time() > self.deadline or local_calls >= max_calls:
                    break
                
                # Sort offspring
                samples.sort(key=lambda x: x[1])
                
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * samples[i][0]
                x_new_mean = y_w
                
                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)
                
                # Check for improvement in best value
                if local_best_f < best_improvement:
                    best_improvement = local_best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if norm_diff < 1e-14 or stagnation_counter > 10:
                    # Stagnation: no movement in mean or no improvement
                    # Reduce sigma and break if stuck too long or just break
                    # For robustness, break to allow restart
                    break
                
                # Update C^{-1/2}
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
                
                # Update evolution paths
                # p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + \
                          math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # h_sigma check
                # Approximation of expected norm
                # More accurate check would use generation count, but this is a robust heuristic
                h_sigma = 1.0
                if norm_p_sigma > 1.4 * chi_n:
                    h_sigma = 0.0
                
                # p_c
                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
                
                # Update sigma
                sigma = sigma * math.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Update C
                c1_term = c_1 * np.outer(p_c, p_c)
                
                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = samples[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)
                
                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2.0
                # Add small regularization to ensure PD if needed
                # C += 1e-10 * np.eye(dim)
                
                x_mean = x_new_mean
            
            return local_best_x, local_best_f

        # Strategy: Multi-start CMA-ES
        # 1. Initial global exploration
        # 2. Local refinement
        
        # Phase 1: Global Exploration
        # Use a few random starts or a large sigma CMA-ES
        # Since dim=5, we can afford a decent global search.
        # Let's do 10 random starts with small budgets, or one large budget with large sigma.
        # A large sigma CMA-ES might get stuck. Multi-start is safer.
        
        n_starts = 5
        calls_per_start = max(10, budget // (n_starts * 2)) # Leave budget for refinement
        # Actually, let's do a balanced approach:
        # 3 random starts with ~50 calls each, then refine best with remaining.
        
        # Even better: One CMA-ES with large sigma for initial phase?
        # Let's try:
        # 1. Start at 0 with sigma=1.0, run for 20% budget.
        # 2. Start at best found with sigma=0.1, run for 80% budget.
        
        # Or multi-start:
        # 5 starts, 40 calls each = 200 calls.
        # Refine best with 300 calls.
        
        # Let's go with a dynamic allocation:
        # Phase 1: 4 random starts + origin, 50 calls each. Total 250 calls.
        # Phase 2: Refine best point with remaining budget (250 calls).
        
        # But what if the optimum is far?
        # Let's use a large sigma for the first start (origin) to cover space.
        
        # Revised Strategy:
        # 1. Run CMA-ES from origin with sigma=2.0 for 100 calls.
        # 2. Run CMA-ES from best found with sigma=0.5 for remaining budget.
        
        # This is simpler and often effective.
        
        # Let's try:
        # 1. Global: Origin, sigma=2.0, 100 calls.
        # 2. Local: Best, sigma=0.1, remaining calls.
        
        # To be robust against hidden shifts/rotations, random starts are better.
        # Let's do:
        # 1. 5 random points in [-5, 5]^5, sigma=1.0, 40 calls each. (200 calls)
        # 2. Refine best with sigma=0.1, remaining calls.
        
        # Let's implement this.
        
        n_global = 5
        budget_global = 40
        total_global_calls = n_global * budget_global
        
        # Ensure we don't exceed budget
        if total_global_calls >= budget:
            # If budget is too small, just do one run
            run_cma_es(np.zeros(dim), 1.0, budget)
            return self.best_x, self.best_f
        
        # Run global starts
        for i in range(n_global):
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            # Random start
            x_start = np.random.uniform(-5.0, 5.0, dim)
            if i == 0:
                x_start = np.zeros(dim) # Always include origin
            
            run_cma_es(x_start, 1.0, budget_global)
        
        # Phase 2: Local Refinement
        remaining_budget = budget - self.calls
        if remaining_budget > 0 and self.best_x is not None:
            # Start from best found
            x_start = self.best_x.copy()
            # Small sigma for local search
            sigma_start = 0.1
            run_cma_es(x_start, sigma_start, remaining_budget)
        
        return self.best_x, self.best_f
