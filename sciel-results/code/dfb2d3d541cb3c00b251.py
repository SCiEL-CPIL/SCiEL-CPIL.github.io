"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start random scan for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, budget, rng, sigma0=1.0, deadline=None):
        """Run CMA-ES from x0 with given budget."""
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))
        lambda_ = pop_size
        
        # Initialize CMA-ES parameters
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        cs = np.zeros(n)
        
        # Evolution paths
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        cmu = (2 + 2 / (n + 1)) / ((n + 4) ** 2 + mueff)
        ccs = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = (2 + 2 / (n + 1.7)) / ((n + 2) ** 2 + mueff)
        cmu = (2 - c1 + 2 * weights[0]) / ((n + 2) ** 2 + mueff)
        
        # Initial step size
        sigma = sigma0
        
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1
        
        # Precompute sqrt(C) for sampling
        def sample_from_C():
            """Sample from N(0, C) using Cholesky."""
            L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            return L @ rng.standard_normal(n)
        
        while calls < budget:
            if deadline is not None and time.time() > deadline:
                break
                
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                    step = L @ z
                except np.linalg.LinAlgError:
                    # Fallback: use eigendecomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    step = eigvecs @ (np.sqrt(eigvals) * z)
                
                y = x + sigma * step
                y = np.clip(y, -5, 5)
                offspring.append(y)
                calls += 1
                if calls >= budget:
                    break
            
            # Evaluate offspring
            f_values = np.array([func(y) for y in offspring])
            
            # Sort by fitness
            order = np.argsort(f_values)
            y_sorted = np.array([offspring[i] for i in order])
            f_sorted = f_values[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Update mean
            w = weights[:min(mu, len(offspring))]
            x_new = np.sum(w[:, None] * y_sorted[:min(mu, len(offspring))], axis=0)
            
            # Update covariance matrix
            ydiff = (y_sorted[:min(mu, len(offspring))] - x) / sigma
            delta_x = x_new - x
            
            # Update evolution paths
            pc = (1 - cmu) * pc + cmu * delta_x / sigma
            cs = (1 - ccs) * cs + np.sqrt(ccs * (2 - ccs) * mueff) * delta_x / sigma
            
            # Update covariance
            C = (1 + c1 * (mueff - 1) * np.outer(pc, pc) / (1 - (1 - cmu) ** 2) 
                 - c1) * C + c1 * np.sum(w[:, None] * np.outer(ydiff, ydiff), axis=0)
            
            # Update step size
            sigma *= np.exp((ccs / (n + 4)) * (np.linalg.norm(cs) / 
                          np.sqrt(1 - (1 - ccs) ** 2) - 1))
            
            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x = x_new
            x = np.clip(x, -5, 5)
            
            # Check for convergence
            if np.linalg.norm(delta_x) < 1e-10:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Phase 1: Random scan to find promising basins
        scan_budget = min(50, budget // 10)
        points = rng.uniform(-5, 5, (scan_budget, n))
        values = np.array([func(p) for p in points])
        
        # Keep top candidates
        top_k = min(5, scan_budget)
        top_indices = np.argsort(values)[:top_k]
        
        best_x = points[top_indices[0]].copy()
        best_f = values[top_indices[0]]
        
        # Phase 2: CMA-ES from best scan point
        remaining = budget - scan_budget
        deadline = time.time() + 280  # 300s limit with 20s margin
        
        # Try CMA-ES from multiple starting points
        for idx in top_indices:
            if remaining <= 0:
                break
            if time.time() > deadline:
                break
            
            # Allocate budget proportionally
            cma_budget = min(remaining, max(20, remaining // (len(top_indices) - idx)))
            x_start = points[idx].copy()
            
            x_opt, f_opt, used = self._cma_es(func, x_start, cma_budget, rng, 
                                               sigma0=1.0, deadline=deadline)
            remaining -= used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Phase 3: Final refinement if budget remains
        if remaining > 10 and time.time() < deadline:
            x_opt, f_opt, used = self._cma_es(func, best_x, remaining, rng, 
                                               sigma0=0.5, deadline=deadline)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
