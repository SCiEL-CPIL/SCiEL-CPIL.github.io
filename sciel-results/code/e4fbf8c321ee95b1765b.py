"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
            return fx

        # CMA-ES Parameters
        # For dim=5, lambda=9 is a good standard choice
        lam = 9
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / weights[0]
        ccm = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        dsigma = (1 + 2 * max(0, sqrt(mueff / dim + 2) - 1)) * (1 + ccm)
        # Note: sqrt is not imported, so we use ** 0.5
        dsigma = (1 + 2 * max(0, (mueff / dim + 2) ** 0.5 - 1)) * (1 + ccm)
        cc = (4 + dim / mueff) / (mueff + 4 + 2 * dim / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        def run_cma_es(x0, sigma0, max_calls):
            nonlocal calls_used, best_x, best_f
            
            # Check if we have enough budget
            if calls_used >= budget:
                return

            # Initialize CMA state
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = (1 + ccm) / (dim + 1)
            ds = (1 + 2 * max(0, (mueff / dim + 2) ** 0.5 - 1)) * (1 + ccm)
            # Cholesky factor of C
            Ch = np.linalg.cholesky(C)
            # Precompute sqrt weights for sampling
            sqrt_weights = np.sqrt(weights)

            # Initial population
            pop = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            # We need to be careful with the call count
            # In standard CMA, we generate lam points per iteration.
            # With budget 500 and lam 9, we can do ~55 iterations.
            
            while calls_used + lam <= budget:
                if time.time() > deadline:
                    return

                # Sample population
                for i in range(lam):
                    y = np.random.normal(0, 1, dim)
                    pop[i] = mean + sigma * (Ch @ y)
                    pop_f[i] = eval_point(pop[i])

                # Sort
                idx = np.argsort(pop_f)
                pop = pop[idx]
                pop_f = pop_f[idx]

                # Update mean
                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # Update sigma
                # ps update
                ps = (1 - dsigma) * ps + dsigma * np.sqrt(mueff) / np.linalg.norm(Ch) * (Ch.T @ (mean - old_mean) / sigma)
                
                # C update
                pc = (1 - ccm) * pc + ccm * (mean - old_mean) / sigma
                
                # Rank-one and rank-mu updates
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - ccm)**2) * np.eye(dim)) + cmu * np.sum(weights[:mu, None, None] * np.outer(pop[:mu] - old_mean, pop[:mu] - old_mean), axis=0) / sigma**2
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    Ch = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add small regularization
                    C += 1e-10 * np.eye(dim)
                    Ch = np.linalg.cholesky(C)

                # Update sigma
                sigma *= np.exp(cc * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                # Convergence check
                if sigma < 1e-12:
                    break

        # Multi-Start Strategy
        # We split the budget among several starts.
        # Let's try 5 starts, each with 100 calls (500 total).
        # Or adaptive: keep going until budget is exhausted.
        # Since CMA-ES is expensive, we limit the number of restarts.
        
        num_starts = 5
        calls_per_start = budget // num_starts
        
        for i in range(num_starts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Random start point
            x0 = np.random.uniform(-4, 4, dim)
            sigma0 = 1.0  # Initial step size
            
            run_cma_es(x0, sigma0, calls_per_start)

        return best_x, best_f
# EVOLVE-BLOCK-END
