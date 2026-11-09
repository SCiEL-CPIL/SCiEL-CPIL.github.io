"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Try multiple restarts with different initializations
        # CMA-ES parameters
        dim = self.dim

        # Initial sigma and step size
        # For f11 (Discus), the landscape is very flat in one direction
        # We need a good initial sigma to explore

        # We'll do a few restarts with different strategies
        # First, let's try a simple but effective approach:
        # 1. Start with a reasonable point
        # 2. Use adaptive step size
        # 3. Use CMA-ES for the final polish

        # Phase 1: Quick exploration with random restarts
        # Phase 2: CMA-ES refinement

        # Let's use a hybrid approach:
        # - Start with several random points
        # - Use (1+lambda)-ES with adaptation
        # - Switch to CMA-ES when close to optimum

        # Actually, let's use a well-tuned (1+1)-ES with better parameters
        # and add restarts based on stagnation

        rng = np.random.default_rng(42)

        # Multiple restarts strategy
        num_restarts = 3
        calls_per_restart = (self.budget - 1) // num_restarts

        for restart in range(num_restarts):
            if calls_used >= self.budget - 1:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                x = np.zeros(dim)
            else:
                # Try different starting points
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            calls_used += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # Track progress for adaptive sigma
            no_improve = 0
            max_no_improve = 30

            while calls_used < self.budget - 1 and no_improve < max_no_improve:
                if time.time() > deadline:
                    break

                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

        # Phase 2: CMA-ES refinement if we have budget left
        if calls_used < self.budget - 1 and best_x is not None:
            if time.time() < deadline - 5:
                # Use CMA-ES for final refinement
                x0 = best_x.copy()
                f0 = best_f
                sigma = 0.5

                # Simple CMA-ES implementation
                dim = self.dim
                pop_size = max(4, int(4 + 3 * np.log(dim)))

                # CMA-ES parameters
                c_sigma = (pop_size + 2) / (dim + pop_size + 5)
                d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
                c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
                c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
                c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) / (dim + 2) - 1))

                # Initialize CMA-ES state
                mean = x0.copy()
                C = np.eye(dim)
                sigma = 0.3
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                cs = (pop_size + 2) / (dim + pop_size + 5)
                ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
                cc = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
                c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
                cmu = min(1 - c1, 2 * ((pop_size / 2 + 1) / (dim + 2) - 1))

                # Eigen decomposition cache
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(eigvals)
                B = eigvecs

                max_iter = (self.budget - 1 - calls_used) // pop_size
                if max_iter < 1:
                    max_iter = 1

                for it in range(max_iter):
                    if calls_used + pop_size > self.budget - 1:
                        break
                    if time.time() > deadline:
                        break

                    # Generate population
                    offspring = np.zeros((pop_size, dim))
                    for i in range(pop_size):
                        z = rng.standard_normal(dim)
                        offspring[i] = mean + sigma * (B @ (D * z))
                        offspring[i] = np.clip(offspring[i], -5, 5)

                    # Evaluate
                    fitnesses = np.zeros(pop_size)
                    for i in range(pop_size):
                        fitnesses[i] = func(offspring[i])
                        calls_used += 1
                        if fitnesses[i] < best_f:
                            best_f = fitnesses[i]
                            best_x = offspring[i].copy()

                    # Sort by fitness
                    sorted_idx = np.argsort(fitnesses)
                    mu = pop_size // 2
                    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    w /= w.sum()
                    mu_eff = 1 / (w @ w)

                    # Update mean
                    mean = (w[:, np.newaxis] * offspring[sorted_idx[:mu]]).sum(axis=0)

                    # Update step size
                    y_sigma = (B @ (D @ pc)) / sigma
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                    # Update covariance
                    y = (offspring[sorted_idx[:mu]] - mean) / sigma
                    delta_sig = np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1)) - 1
                    rank_one = np.outer(ps, ps)
                    rank_mu = (w[:, np.newaxis] * y).T @ y
                    C = (1 - c_c * cc * c_sigma - c1 - cmu) * C + c_c * cc * c_sigma * rank_one + c1 * rank_one + cmu * rank_mu

                    # Update eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    B = eigvecs

                    # Check convergence
                    if np.linalg.norm(mean - x0) < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
