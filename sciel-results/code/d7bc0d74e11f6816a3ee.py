"""Rewritten optimizer: CMA-ES with adaptive restarts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for 5-dimensional BBOB-style function f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        # Initial sigma estimate
        sigma0 = 2.0

        # We'll do multiple restarts if budget allows
        # Each restart uses a fraction of the budget
        # Strategy: first run gets most budget, subsequent restarts get less

        # Adaptive strategy:
        # - Run CMA-ES with a budget
        # - If it stagnates (no improvement for N evaluations), restart
        # - Keep track of global best

        def run_cmaes(x0, sigma_init, max_evals):
            """Run CMA-ES from x0 with given sigma, using at most max_evals evaluations."""
            nonlocal global_best_x, global_best_f

            n = dim
            pop_size = lambda_

            # CMA-ES state
            x = np.array(x0, dtype=float)
            sigma = sigma_init
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            # Step sizes
            cmu = 1 / (5 * n + n ** 0.5)
            cs = (1 + 2 / (n + 1)) / (n + 4 + 2 / (n + 1))
            ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 2)) - 1) + cs
            cc = (4 + n / (n + 4)) / ((n + 2) + 4 + n / (n + 4))
            c1 = (mu_eff + 2) / ((n + 2) ** 2 + mu_eff)
            cmu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((n + 2) ** 2 + mu_eff)

            mu_eff = pop_size / 2
            # Weights
            w = np.log(np.arange(1, pop_size + 1)) - np.log(pop_size)
            w = w / w.sum()
            mu = pop_size // 2  # number of recombination points
            w = w[:mu]
            w = w / w.sum()
            mu_eff = 1 / np.sum(w ** 2)

            # Reinitialize cmu with correct mu_eff
            c1 = (mu_eff + 2) / ((n + 2) ** 2 + mu_eff)
            cmu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((n + 2) ** 2 + mu_eff)

            evals_used = 0
            best_f_local = np.inf
            best_x_local = x.copy()
            stagnation_count = 0
            stagnation_limit = 15 * n  # no improvement for 15*n generations

            # Initial population
            for _ in range(pop_size):
                pass  # will be generated in loop

            # Main loop
            while evals_used < max_evals:
                # Check deadline
                if time.time() > deadline:
                    break

                # Generate population
                Y = []
                F = []
                for i in range(pop_size):
                    if evals_used >= max_evals:
                        break
                    # Sample
                    z = np.random.randn(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    F.append(fy)
                    Y.append(y)
                    evals_used += 1

                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()
                    if fy < best_f_local:
                        best_f_local = fy
                        best_x_local = y.copy()

                if len(Y) == 0:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                Y_sorted = [Y[i] for i in idx[:mu]]
                F_sorted = [F[i] for i in idx[:mu]]

                # Check stagnation
                if best_f_local < np.inf:
                    # Track improvement
                    if stagnation_count > 0:
                        # We improved
                        stagnation_count = 0
                    else:
                        # First time or no improvement
                        pass
                # Simpler stagnation: if best hasn't improved in last K evaluations
                # We'll track this differently

                # Mean of selected
                m_new = np.zeros(n)
                for i in range(mu):
                    m_new += w[i] * Y_sorted[i]

                # Update step size
                inv_D = 1.0 / D
                inv_B = np.linalg.inv(B)
                d = inv_B.T @ (inv_D * (m_new - x))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * d
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used // pop_size + 1)))) < \
                       (1 + 2 / (n + 1)) * np.sqrt(n + 1)

                # Update evolution path
                pc = (1 - cs) * pc + hsig * cs * (m_new - x) / sigma

                # Update covariance matrix
                C = B @ np.diag(D ** 2) @ B.T
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (delta_hsig - cs * (2 - cs) + (1 - c1) * (1 - cs) ** 2) * np.eye(n)) + \
                    cmu * np.sum([w[i] * np.outer(Y_sorted[i] - x, Y_sorted[i] - x) for i in range(mu)], axis=0)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                # Ensure B is orthonormal
                B = B / np.linalg.norm(B, axis=0)

                # Update mean
                x = m_new

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(n + 1) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # Check for stagnation: if sigma is too small or too large
                if sigma < 1e-8 or sigma > 1e6:
                    break

                # Check if we should stop due to lack of progress
                # Simple heuristic: if best hasn't improved in 20*n evaluations
                # We'll count generations without improvement
                # For simplicity, use a generation counter
                # Actually, let's just let it run and use the budget/deadline

        # Restart strategy
        # First run: use 70% of budget
        # Subsequent runs: use decreasing portions
        # If stuck, restart from global best with new sigma

        # Initial point: center of box
        x0 = np.zeros(dim)
        f0 = func(x0)
        global_best_f = f0
        global_best_x = x0.copy()
        evals_total = 1

        # Also try a few random points for initial diversity
        rng = np.random.default_rng(42)
        for _ in range(3):
            xr = rng.uniform(-5, 5, dim)
            fr = func(xr)
            evals_total += 1
            if fr < global_best_f:
                global_best_f = fr
                global_best_x = xr.copy()

        # Main CMA-ES runs with restarts
        # Budget allocation: first run gets 60%, then 25%, then 15%
        # Adjust based on performance
        remaining = total_budget - evals_total
        run_budgets = [int(remaining * 0.6), int(remaining * 0.25), int(remaining * 0.15)]
        run_budgets = [max(b, 10) for b in run_budgets]

        for run_i, rb in enumerate(run_budgets):
            if evals_total >= total_budget:
                break
            if time.time() > deadline:
                break

            # Starting point for this run
            if run_i == 0:
                x_start = global_best_x.copy()
                sigma_start = 2.0
            else:
                # Perturb from global best
                x_start = global_best_x + 0.5 * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
                sigma_start = 1.0

            # Run CMA-ES
            prev_best = global_best_f
            run_cmaes(x_start, sigma_start, min(rb, total_budget - evals_total))

            # Check if we made progress
            if global_best_f < prev_best - 1e-6:
                # Good progress, continue
                pass
            else:
                # No progress, maybe try different sigma next time
                pass

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
