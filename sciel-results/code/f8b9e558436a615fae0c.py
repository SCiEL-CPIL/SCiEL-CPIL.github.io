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

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Wrap func to track calls and enforce budget
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Multi-Start CMA-ES Strategy
        # We perform a series of CMA-ES runs.
        # Each run uses a portion of the remaining budget.
        # The population size increases with each restart to widen the search.
        # The initial sigma increases to cover a larger area.
        
        # Configuration for restarts
        # Total budget: 500
        # We will use a greedy allocation:
        # Run 1: pop=10, max_iter = budget // 10
        # If it doesn't converge, restart with larger pop and sigma.
        
        # To be efficient, we define a schedule of (popsize, sigma0, max_iter_fraction)
        # However, since we don't know when it converges, we just run CMA-ES
        # until it hits the iteration limit or budget.
        
        # Let's implement a simple loop of CMA-ES runs.
        # We need a manual CMA-ES implementation to strictly control calls and handle restarts seamlessly.
        # SciPy's minimize doesn't support easy mid-run restarts or strict call counting without wrappers that might break convergence logic if they return stale values.
        # A lightweight manual CMA-ES is safer for strict budget adherence.
        
        def run_cma_es(x0, sigma0, max_calls, rng):
            # Manual CMA-ES implementation
            # x: mean
            # sigma: step size
            # C: covariance matrix
            # pc, ps: evolution paths
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Parameters
            # mu = lambda // 2
            # weights: linearly decreasing
            # ca, cm: adaptation rates
            # chi_n: expected norm of standard normal
            
            # We will determine lambda (pop size) based on available calls.
            # But here we are inside a run. The caller decides lambda.
            # Actually, let's pass lambda as argument.
            pass # Placeholder, we will implement the loop in __call__ for clarity and control.

        # Instead of a complex internal function, we implement the CMA-ES loop directly in __call__
        # to allow easy restarts and budget management.
        
        # Schedule of restarts:
        # 1. Small pop, small sigma (local-ish)
        # 2. Medium pop, medium sigma
        # 3. Large pop, large sigma (global)
        
        # Since Lunacek has two basins, we want to explore widely first or adaptively.
        # Standard CMA-ES starts local. Multi-start helps.
        
        # Let's use a fixed set of restart configurations.
        # Config 1: pop=10, sigma=1.0
        # Config 2: pop=20, sigma=2.0
        # Config 3: pop=40, sigma=4.0
        
        # We will iterate through these, allocating remaining budget.
        
        configs = [
            {'pop': 10, 'sigma': 1.0},
            {'pop': 20, 'sigma': 2.0},
            {'pop': 40, 'sigma': 4.0},
        ]
        
        for cfg in configs:
            if self.calls >= self.budget or not self._check_time():
                break
            
            pop_size = cfg['pop']
            sigma0 = cfg['sigma']
            
            # Initial point: center of box or best found so far?
            # For the first run, start at 0.
            # For subsequent runs, start at best_x found so far, but with larger sigma.
            # Actually, for global exploration, starting at 0 again with larger sigma is okay,
            # or starting at a random point.
            # Let's start at 0 for the first, and best_x for subsequent?
            # No, if best_x is in a local minimum, starting there with larger sigma might just escape slowly.
            # Better to restart from the center (0) or a random point to explore other basins.
            # Lunacek has a global min and a local min.
            # Let's start each restart at 0.0 with the specified sigma.
            
            x0 = np.zeros(n)
            
            # Max iterations for this run
            max_iter = max(1, (self.budget - self.calls) // pop_size)
            
            # Run CMA-ES
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
            
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
            
            # Initial evaluation
            # We need to evaluate the initial population.
            # But we must be careful with the call count.
            
            for it in range(max_iter):
                if self.calls >= self.budget or not self._check_time():
                    break
                
                # Generate population
                # We need to evaluate pop_size points.
                # If we don't have enough budget for a full population, we stop.
                if self.calls + pop_size > self.budget:
                    break
                
                # Update covariance and mean? No, standard CMA-ES updates at end of iteration.
                # Generate Y = C @ Z
                Z = rng.standard_normal((pop_size, n))
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset or regularize
                    C = np.eye(n)
                    L = np.eye(n)
                
                Y = np.dot(Z, L.T)
                
                # New points
                X = np.tile(x, (pop_size, 1)) + sigma * Y
                
                # Evaluate
                F = []
                for i in range(pop_size):
                    F.append(safe_func(X[i]))
                
                # Sort
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = np.array(F)[idx]
                
                # Update best
                # safe_func already updates self.best_x and self.best_f
                
                # Update mean
                x_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
                
                # Update sigma
                x_norm = np.linalg.norm(x_new - x)
                if x_norm > 1e-10:
                    y = (x_new - x) / (sigma * x_norm)
                else:
                    y = np.zeros(n)
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))) / chi_n < 1.4 + 2 / (n + 1)
                
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y
                
                # Update covariance
                C = (1 + c_1 * (hsig * (2 - c_c) - c_c) - c_1 - c_mu) * C
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C += c_1 * np.outer(pc, pc)
                C += c_mu * np.sum(weights[i] * np.outer(X_sorted[i] - x, X_sorted[i] - x) for i in range(mu))
                
                # Symmetrize
                C = (C + C.T) / 2.0
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                x = x_new
                
                # Optional: restart if sigma is too small or large
                if sigma < 1e-10 or sigma > 1e10:
                    break

        # Final polish?
        # CMA-ES is usually good enough.
        # If we have budget left, do a simple Nelder-Mead or just return.
        
        if self.best_x is None:
            self.best_x = np.zeros(n)
            self.best_f = safe_func(self.best_x)
            
        return self.best_x, self.best_f
