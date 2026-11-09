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

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def _check_deadline():
            return time.time() > deadline
        
        # Phase 1: Latin Hypercube Sampling spread (cheap global exploration)
        # This gives us a good initial best quickly for AOCC
        n_lhs = 31  # measured: 31 points is a good balance
        lhs_samples = self._lhs_sample(n_lhs, dim)
        
        for x in lhs_samples:
            if _check_deadline():
                break
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from best sample (rotation-adaptive)
        # Allocate ~70% of remaining budget to CMA-ES
        remaining = budget - call_count
        if remaining > 20 and not _check_deadline():
            cma_budget = int(remaining * 0.70)
            cma_budget = max(cma_budget, 10)
            cma_budget = min(cma_budget, remaining)
            
            best_x, best_f, calls_used = self._cma_es(
                func, best_x, best_f, cma_budget, dim, 
                deadline=deadline, start_calls=call_count
            )
            call_count += calls_used
        
        # Phase 3: Final L-BFGS-B polish (tight tolerance for extra precision)
        if not _check_deadline() and best_x is not None:
            from scipy.optimize import minimize
            try:
                result = minimize(
                    func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                # Count function calls from L-BFGS-B
                nf = result.get('nfev', 0)
                call_count += nf
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass
        
        return best_x, best_f

    def _lhs_sample(self, n, dim):
        """Generate Latin Hypercube samples in [-5, 5]^dim."""
        rng = np.random.default_rng(42)
        samples = np.zeros((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.random(n)) / n
            samples[:, j] = -5 + 10 * samples[:, j]
        return samples

    def _cma_es(self, func, x0, f0, budget, dim, deadline, start_calls=0):
        """Simple CMA-ES implementation."""
        rng = np.random.default_rng(123)
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma = 2.0
        x = x0.copy()
        f_best = f0
        calls = 0
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Update rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma1 = c_sigma
        c_sigma1 = (mu_eff + 2) / (dim + mu_eff + 5)
        c_sigma1 = (1 + 2 * (mu_eff - 1) / (dim + 1)) / (dim + 3 + 2 * (mu_eff - 1) / (dim + 1))
        c_sigma1 = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1))) / (dim + 3 + 2 * max(0, (mu_eff - 1) / (dim + 1)))
        
        # Simplified update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma1 = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1))) / (dim + 3 + 2 * max(0, (mu_eff - 1) / (dim + 1)))
        c_sigma1 = c_sigma1 / (1 + c_sigma)
        
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        
        # Eigenvalue decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        
        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate population
            offspring = []
            for i in range(pop_size):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sqrt(eigvals)) @ z
                y = eigvecs @ (np.sqrt(eigvals) * z)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                offspring.append((fi, xi))
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            for fi, xi in offspring[:mu]:
                if fi < f_best:
                    f_best = fi
                    x = xi.copy()
            
            # Compute weighted mean of best mu
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i][1]
            
            # Update evolution paths
            # Need C^(-1/2)
            c_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            delta = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * c_inv_sqrt @ delta
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                   (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
            rank_mu = np.outer(x_new - x, x_new - x) / (sigma ** 2)
            rank_1 = np.outer(pc, pc)
            
            C = (1 + c_sigma1 * (2 - c_sigma1) * (hsig * np.outer(ps, ps) - C) + 
                 c_sigma1 * (2 - c_sigma1) * (hsig * np.outer(ps, ps) - C) +
                 c_c * (2 - c_c) * (rank_1 - C) + 
                 c_sigma1 * (2 - c_sigma1) * (rank_mu - C))
            
            # Simplified: just use rank update
            C = (1 - c_sigma1 * (2 - c_sigma1) - c_c * (2 - c_c)) * C + \
                c_sigma1 * (2 - c_sigma1) * (hsig * np.outer(ps, ps) + rank_mu) + \
                c_c * (2 - c_c) * (rank_1)
            
            # Ensure positive definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * 
                           (np.linalg.norm(ps) / np.sqrt((1 - (1 - c_sigma) ** (2 * (gen + 1)))) - 
                            np.sqrt((mu_eff + 1) / (dim + 1))))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x = x_new
        
        return x, f_best, calls
# EVOLVE-BLOCK-END
