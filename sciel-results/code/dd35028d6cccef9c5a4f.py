"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random initialization and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 0: Random Initialization ---
        # Sample a few points to find a good starting basin
        n_init = 10
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        calls_used = n_init

        # --- Phase 1: CMA-ES ---
        x_mean = best_x.copy()
        sigma = 1.0  # Start with a moderate step size
        
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # CMA-ES parameters
        cmac = 1.0 / (dim + 4)
        cmam = (4 + dim / 4) / (dim + 4 + 3)
        c_sigma = (dim + 1.3) / (dim + 5.3 + 2 * lam / mu)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Reserve budget for polish
        polish_budget = 50
        cma_budget = budget - polish_budget
        
        # We already used n_init calls
        # Max calls in CMA loop: cma_budget - n_init? No, cma_budget is total allowed for CMA phase including init?
        # Let's say CMA phase can use up to budget - polish_budget calls total.
        # So remaining for CMA loop is (budget - polish_budget) - n_init
        
        max_cma_calls = (budget - polish_budget) - n_init
        if max_cma_calls < 0:
            max_cma_calls = 0
            
        calls_in_cma = 0

        while calls_in_cma < max_cma_calls:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            fitness = np.zeros(lam)

            for i in range(lam):
                if calls_in_cma >= max_cma_calls:
                    break
                if time.time() > deadline - 5.0:
                    break

                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring[i] = y
                fitness[i] = func(y)
                calls_in_cma += 1

                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = y.copy()

            if calls_in_cma >= max_cma_calls:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            offspring = offspring[idx]
            fitness = fitness[idx]

            # Weighted recombination
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i]

            # Update mean
            v = (x_new - x_mean) / sigma
            x_mean = x_new

            # Update covariance matrix
            pc = (1 - cmac) * pc + cmac * v
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cmac) ** 2) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * v / chi_n

            delta_h = (1 - hsig) * c_sigma * (2 - c_sigma)
            C = ((1 - cmac * (2 - cmac) + cmac * cmam * delta_h) * C
                 + (cmac * (2 - cmac) - cmam * delta_h) * np.outer(pc, pc)
                 + c_sigma * (2 - c_sigma) * mueff * np.outer(ps, ps))

            # Ensure positive definite
            C = (C + C.T) / 2
            # Eigendecomposition for step size
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

        # --- Phase 2: (1+1)-ES Polish ---
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        remaining_budget = budget - calls_used
        
        # Use all remaining budget for polish
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.85

        return x, fx
# EVOLVE-BLOCK-END
