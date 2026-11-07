"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda
        c1 = (2 / ((n + 2) ** 2 + 10))
        cmu = min(0.5 - c1, 2 / ((n + 1) ** 2 + 5))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        ds = 1 + 2 * max(0, np.sqrt((cs + 1) / (cs + 1) - 1))
        cc = (4 + n / n) / (4 + n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial point and sigma
        rng = np.random.default_rng(42)
        x_mean = np.zeros(n)
        sigma = 2.0

        # Evaluation budget tracking
        calls_made = 0
        best_x = None
        best_f = np.inf

        # CMA-ES state
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cov = np.eye(n)
        ev = np.linalg.eigh(cov)
        D = np.sqrt(ev[0])
        B = ev[1]

        for gen in range(budget // pop_size + 1):
            if time.time() > deadline - 2.0:
                break
            if calls_made >= budget:
                break

            # Generate population
            offspring = np.zeros((pop_size, n))
            for i in range(pop_size):
                if time.time() > deadline - 1.0:
                    break
                if calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                offspring[i] = x_mean + sigma * (B @ (D * z))
                offspring[i] = np.clip(offspring[i], -5, 5)
                f_val = func(offspring[i])
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = offspring[i].copy()

            # Sort by fitness
            fitnesses = np.array([func(offspring[i]) if i < pop_size else np.inf for i in range(pop_size)])
            # Actually, we already evaluated them, so let's store them properly
            # Let me restructure - we need to store fitnesses during generation
            
            # Re-do: store fitnesses during generation
            # Let's restart this section properly
            
            break  # This first loop structure is flawed, let me rewrite

        # Let me rewrite the entire __call__ properly
        return best_x, best_f

    def _cmaes(self, func, n, budget, deadline):
        """CMA-ES implementation."""
        pop_size = max(4, int(4 + 3 * np.log(n)))
        c1 = (2 / ((n + 2) ** 2 + 10))
        cmu = min(0.5 - c1, 2 / ((n + 1) ** 2 + 5))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        ds = 1 + 2 * max(0, np.sqrt((cs + 1) / (cs + 1) - 1))
        cc = (4 + n / n) / (4 + n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        rng = np.random.default_rng(42)
        x_mean = np.zeros(n)
        sigma = 2.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cov = np.eye(n)

        ev = np.linalg.eigh(cov)
        D = np.sqrt(np.maximum(ev[0], 1e-20))
        B = ev[1]

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        while calls_made < budget:
            if time.time() > deadline - 2.0:
                break

            # Generate offspring
            offspring = np.zeros((pop_size, n))
            fitnesses = np.zeros(pop_size)

            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 1.0:
                    break
                z = rng.standard_normal(n)
                x_new = x_mean + sigma * (B @ (D * z))
                x_new = np.clip(x_new, -5, 5)
                offspring[i] = x_new
                fitnesses[i] = func(x_new)
                calls_made += 1
                if fitnesses[i] < best_f:
                    best_f = fitnesses[i]
                    best_x = x_new.copy()

            if calls_made < budget:
                # Sort
                order = np.argsort(fitnesses)
                y_w = np.zeros((mu, n))
                for i in range(mu):
                    y_w[i] = offspring[order[i]]

                # Update mean
                x_mean_new = x_mean + sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update step size control
                y_sigma = (x_mean_new - x_mean) / (sigma + 1e-30)
                ps = np.sqrt(ds) * ps + np.sqrt(1 - ds) * y_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made / pop_size + 1)) / chi_n < 1.4 + 2 / (n + 1))

                # Update evolution paths
                pc = (1 - cs) * pc + hsig * cs * y_sigma
                cov = (1 - (mu_eff_c1 + mu_eff_cmu) * (1 + 2 * c1 / (n + 1))) * cov  # approximate
                # Actually, let's use a simpler covariance update
                delta_hsig = (1 - hsig) * cs * (1 - cs)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y_w[i], y_w[i])

                cov = (1 - (c1 + cmu) * (1 + 2 * c1 / (n + 1))) * cov + \
                      (1 - delta_hsig) * c1 * rank_one + cmu * rank_mu

                # Ensure positive semi-definite
                ev = np.linalg.eigh(cov)
                D = np.sqrt(np.maximum(ev[0], 1e-20))
                B = ev[1]

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x_mean = x_mean_new

        return best_x, best_f, calls_made


class Optimizer:
    """CMA-ES optimizer with time guard for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        c1 = (2 / ((n + 2) ** 2 + 10))
        cmu = min(0.5 - c1, 2 / ((n + 1) ** 2 + 5))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        ds = 1 + 2 * max(0, np.sqrt((cs + 1) / (cs + 1) - 1))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        rng = np.random.default_rng(42)
        x_mean = np.zeros(n)
        sigma = 2.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cov = np.eye(n)

        ev = np.linalg.eigh(cov)
        D = np.sqrt(np.maximum(ev[0], 1e-20))
        B = ev[1]

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        while calls_made < budget:
            if time.time() > deadline - 2.0:
                break

            # Generate offspring
            offspring = np.zeros((pop_size, n))
            fitnesses = np.zeros(pop_size)

            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 1.0:
                    break
                z = rng.standard_normal(n)
                x_new = x_mean + sigma * (B @ (D * z))
                x_new = np.clip(x_new, -5, 5)
                offspring[i] = x_new
                fitnesses[i] = func(x_new)
                calls_made += 1
                if fitnesses[i] < best_f:
                    best_f = fitnesses[i]
                    best_x = x_new.copy()

            if calls_made >= budget or time.time() > deadline - 1.0:
                break

            # Sort by fitness
            order = np.argsort(fitnesses)
            y_w = np.array([offspring[order[i]] for i in range(mu)])

            # Update mean
            x_mean_new = x_mean + sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            x_mean_new = np.clip(x_mean_new, -5, 5)

            # Update step size control
            y_sigma = (x_mean_new - x_mean) / (sigma + 1e-30)
            ps = np.sqrt(ds) * ps + np.sqrt(1 - ds) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made / pop_size + 1)) / chi_n < 1.4 + 2 / (n + 1))

            # Update evolution paths
            pc = (1 - cs) * pc + hsig * cs * y_sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (1 - cs)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y_w[i], y_w[i])

            cov = (1 - (c1 + cmu) * (1 + 2 * c1 / (n + 1))) * cov + \
                  (1 - delta_hsig) * c1 * rank_one + cmu * rank_mu

            # Ensure positive semi-definite
            ev = np.linalg.eigh(cov)
            D = np.sqrt(np.maximum(ev[0], 1e-20))
            B = ev[1]

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x_mean = x_mean_new

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
