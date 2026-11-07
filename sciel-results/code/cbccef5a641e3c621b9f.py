import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline - 1:
                return np.inf
            call_count += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                fx = float(fx)
                if np.isfinite(fx):
                    if fx < best_f:
                        best_f = fx
                        best_x = np.asarray(x, dtype=float).copy()
                    return fx
                return np.inf
            except Exception:
                return np.inf

        # Phase 1: Initial Exploration
        # Sample 20 random points in [-5, 5]^dim to find a good starting point
        n_init = 20
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
            
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: CMA-ES
        # Use CMA-ES for the majority of the budget
        # Let's allocate ~350 evaluations to CMA-ES
        cma_budget = 350
        if call_count + cma_budget > budget:
            cma_budget = budget - call_count
            
        if cma_budget > 10:
            self._run_cma_es(func, safe_call, best_x, best_f, cma_budget, deadline)
            # Note: safe_call updates best_x and best_f via nonlocal closure

        # Phase 3: Local Refinement
        # Use remaining budget for (1+1)-ES or similar local search
        if call_count < budget and best_x is not None:
            self._run_local_search(func, safe_call, best_x, best_f, budget - call_count, deadline)

        return best_x, best_f

    def _run_cma_es(self, func, safe_call, x0, f0, budget, deadline):
        dim = self.dim
        rng = np.random.default_rng(123)
        
        # CMA-ES parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (2 + mueff / dim) / (dim + 2 + mueff / dim)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - c_c - c_sigma)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # State
        x_mean = x0.copy()
        sigma = 0.5  # Initial step size, adjusted based on problem scale
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Track best found in CMA-ES
        cma_best_x = x_mean.copy()
        cma_best_f = f0
        
        # Main loop
        while True:
            if time.time() > deadline - 5:
                break
            # Check budget: we need lambda_ evaluations
            # We don't track call_count here directly, but safe_call does.
            # We need to know if we have enough budget left.
            # Since safe_call increments call_count, we can't easily check "remaining" without accessing nonlocal.
            # Let's pass a flag or check inside safe_call? No, safe_call is shared.
            # Better: The caller (Optimizer.__call__) allocated a specific budget.
            # We should stop if we've used too many.
            # Let's assume the caller ensures we don't exceed total budget.
            # To be safe, we'll just run until time runs out or a reasonable number of iterations.
            # Max iterations = budget // lambda_
            # Let's just run until deadline or convergence.
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                if time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (np.linalg.cholesky(C) @ z)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                offspring.append(y)
                offspring_f.append(fy)
                
            if len(offspring) < 2:
                break
                
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            
            # Update best
            min_idx = np.argmin(offspring_f)
            if offspring_f[min_idx] < cma_best_f:
                cma_best_f = offspring_f[min_idx]
                cma_best_x = offspring[min_idx].copy()
                
            # Sort
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]
            
            # Update mean
            new_mean = np.sum(weights[:mu, np.newaxis] * offspring[:mu], axis=0)
            
            # Update evolution paths
            y_diff = (new_mean - x_mean) / (sigma + 1e-300)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                np.sign(np.linalg.norm(pc)) * np.linalg.norm(y_diff)
            )
            
            # Update covariance matrix
            delta_hsig = (1 + 2 * max(0, np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c)) * np.eye(dim)) + \
                c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(offspring[:mu] - x_mean, offspring[:mu] - x_mean), axis=0) / (sigma ** 2 + 1e-300)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
            sigma = max(1e-6, min(sigma, 10.0))
            
            # Update mean
            x_mean = new_mean
            
            # Check for convergence/stagnation
            # If sigma is very small or C is ill-conditioned, break
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10 or np.max(eigvals) / np.min(eigvals) > 1e10:
                # Regularize or break
                C = np.eye(dim) * 1e-5
                sigma = 0.1
                if time.time() > deadline - 20:
                    break

    def _run_local_search(self, func, safe_call, x0, f0, budget, deadline):
        dim = self.dim
        rng = np.random.default_rng(456)
        x = x0.copy()
        fx = f0
        sigma = 0.1
        
        for _ in range(budget):
            if time.time() > deadline - 2:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = safe_call(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.8
            if fx < self._last_best_f: # Need to update global best
                 pass # safe_call already updated global best_x/best_f
