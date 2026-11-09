"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with initial random sampling for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Initial random sampling to find a good basin
        # Use ~15 random points for quick exploration
        n_init = min(15, budget // 10)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is None:
                break

        # Phase 2: CMA-ES from the best point found
        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Initial state
        x_mean = best_x.copy()
        sigma = 2.0  # initial step size
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Learning rates
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
        cmu = min(cmu, 2 * (1 - c1))

        # Track for stagnation detection
        best_f_history = []
        stagnation_window = 30  # generations without improvement triggers restart
        last_improvement_gen = 0
        gen_count = 0

        while calls_used < budget:
            if time.time() > deadline:
                break

            gen_count += 1

            # Check stagnation: if no improvement in stagnation_window generations
            if gen_count - last_improvement_gen > stagnation_window:
                # Restart with smaller sigma and slightly perturbed mean
                sigma = max(sigma * 0.5, 0.5)
                x_mean = best_x + rng.normal(0, 0.5, n)
                x_mean = np.clip(x_mean, -5, 5)
                C = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)
                last_improvement_gen = gen_count
                continue

            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            offspring = []
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * L @ z
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = evaluate(y)
                if fy is None:
                    break
                offspring_f.append((fy, y))

            if len(offspring_f) < 2:
                break

            # Sort by fitness
            offspring_f.sort(key=lambda x: x[0])

            # Check if we improved
            if offspring_f[0][0] < best_f - 1e-10:
                last_improvement_gen = gen_count

            # Select mu best
            selected = [x[1] for x in offspring_f[:min(mu, len(offspring_f))]]
            if len(selected) < mu:
                break

            # Update mean
            m_old = x_mean.copy()
            x_mean_new = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

            # Update ps (sigma path)
            C_inv_sqrt = np.linalg.inv(L)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean_new - m_old) / sigma

            # Check hsig
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * gen_count / (n * damps))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update pc (covariance path)
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (x_mean_new - m_old) / sigma

            # Update C
            delta_sig = 1.0 if hsig else 0.0
            C = (1 - c1 - cmu + c1 * delta_sig) * C
            C += c1 * (1 - delta_sig) * np.outer(ps, ps) + c1 * delta_sig * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * np.outer(pc, pc))
            C += cmu * np.sum([w[i] * np.outer(selected[i] - m_old, selected[i] - m_old) for i in range(mu)], axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(n)

            x_mean = x_mean_new

            # Adapt sigma
            sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))
            sigma = max(1e-10, min(sigma, 10.0))

            # Check convergence
            if np.linalg.norm(x_mean_new - m_old) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
