"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: LHS spread + CMA-ES + L-BFGS-B polish.
    
    Designed for BBOB-style Sphere function with hidden shift/rotation/offset.
    AOCC rewards early progress, so we start with a cheap global spread.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0  # Map to [-5, 5]

    def _cmaes(self, func, x0, sigma, budget, deadline, dim, call_count):
        """Simple CMA-ES implementation."""
        rng = np.random.default_rng(42)
        x = x0.copy()
        fx = func(x)
        C = np.eye(dim)
        step_size = sigma
        n = dim
        popsize = max(4 + int(3 * np.log(n)), 6)
        
        # CMA-ES parameters
        cmu = 1. / (4 * n + 4)
        ccs = (4 + n / 4) / (n + 4 + 3)
        c1 = 2 / ((n + 1.3) ** 2 + 4)
        cmu = min(cmu, 2 * (1 - c1) / (n ** 2 + 1))
        dsigma = 1 + cmu + 2 * cmu / n
        hsig = (1.4 + 2. / (n + 1)) * np.sqrt((1 + cmu) / (1 - cmu))
        
        # Initialize state
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))
        
        for iteration in range(budget // popsize):
            if time.time() > deadline or call_count[0] >= budget:
                break
                
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(popsize):
                if call_count[0] >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + step_size * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) == 0:
                break
            
            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)
            
            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]
            
            # Update best
            if offspring_f[0] < fx:
                x = offspring[0].copy()
                fx = offspring_f[0]
            
            # Update covariance matrix and step size
            w = np.log(popsize / 2) - np.arange(popsize / 2)
            w = np.concatenate([w, w[::-1]])[:popsize]
            w = w / w.sum()
            mu_eff = 1 / np.sum(w[:int(popsize / 2)] ** 2)
            
            y_mean = np.zeros(n)
            for i in range(int(popsize / 2)):
                y_mean += w[i] * (offspring[i] - x) / step_size
            
            p_sigma = (1 - cmu) * p_sigma + cmu * y_mean
            h = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cmu) ** (2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * chi_n)
            p_c = (1 - ccs) * p_c + h * ccs * y_mean
            
            C = (1 + c1 * (2 - c1) + cmu) * C + \
                c1 * (np.outer(p_c, p_c) - C) + \
                cmu / mu_eff * sum(w[i] * np.outer(y_i, y_i) for i, y_i in enumerate([np.linalg.solve(C_sqrt, (offspring[i] - x) / step_size) for i in range(int(popsize / 2))]))
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Adapt step size
            step_size *= np.exp(csigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            csigma = (1 + cmu / 2) / (n + 3)
            step_size = np.clip(step_size, 1e-10, 10.0)
            
            # Update mean
            x = x + step_size * C_sqrt @ y_mean
        
        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = [0]
        
        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))
        
        # Phase 1: Latin Hypercube Sampling spread (cheap global exploration)
        # This gives us a good initial best quickly for AOCC
        n_lhs = 31
        rng = np.random.default_rng(42)
        lhs_samples = self._lhs_sample(n_lhs, dim, rng)
        
        for x in lhs_samples:
            if time.time() > deadline or call_count[0] >= budget:
                break
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
        
        # Phase 2: CMA-ES from best sample (rotation-adaptive)
        if time.time() < deadline - 30 and call_count[0] < budget - 50:
            try:
                x_cma, f_cma = self._cmaes(wrapped, best_x, 2.0, budget - call_count[0], deadline - 30, dim, call_count)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma
            except Exception:
                pass
        
        # Phase 3: L-BFGS-B final polish (convex Sphere function)
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 50, "ftol": 1e-12, "gtol": 1e-8})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
