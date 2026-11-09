"""CMA-ES with adaptive restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    CMA-ES based optimizer with adaptive restarts for multimodal functions.
    Uses a population-based evolution strategy with covariance adaptation,
    combined with L-BFGS-B local refinement on the best candidates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        rng = np.random.default_rng(42)

        # --- CMA-ES Implementation ---
        def cma_es(x0, sigma0, pop_size, max_calls, sigma_min=1e-10, sigma_max=10.0):
            """
            Run CMA-ES from x0. Returns best_x, best_f, calls_used.
            """
            nonlocal n_calls, best_x, best_f

            x = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            calls_used = 0
            N = dim

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (N + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
            c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
            c_1 = 2 / ((N + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
            chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

            # State variables
            pc = np.zeros(N)
            ps = np.zeros(N)
            C = np.eye(N)
            stepsize = np.sqrt(np.trace(C))

            # Initial population
            best_local_x = x.copy()
            best_local_f = np.inf

            for gen in range(max_calls // pop_size + 1):
                if n_calls >= budget or check_time():
                    break

                # Generate population
                population = []
                fitnesses = []
                for i in range(pop_size):
                    if n_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(N)
                    y = x + sigma * z
                    fy, y_eval = call_func(y)
                    if fy is None:
                        continue
                    population.append(y_eval)
                    fitnesses.append(fy)
                    calls_used += 1

                if len(population) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fitnesses)
                population = [population[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]

                # Update best
                for i in range(min(mu, len(population))):
                    if fitnesses[i] < best_local_f:
                        best_local_f = fitnesses[i]
                        best_local_x = population[i].copy()
                        if fitnesses[i] < best_f:
                            best_f = fitnesses[i]
                            best_x = population[i].copy()

                # Mean update
                x_new = np.zeros(N)
                for i in range(mu):
                    x_new += weights[i] * population[i]
                x = x_new

                # Step size control (CSA)
                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(N)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_N * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1)))) < (1.4 + 2 / (N + 1)) * chi_N
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

                # Covariance matrix update
                delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1)) * c_sigma
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((N, N))
                for i in range(mu):
                    zi = (population[i] - x_new) / sigma if sigma > 0 else np.zeros(N)
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * (c_c * (2 - c_c) * np.eye(N) - C)) + c_mu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization
                C += 1e-12 * np.eye(N)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
                sigma = max(sigma_min, min(sigma_max, sigma))

                # Check for convergence
                if sigma < sigma_min or np.all(np.abs(C) < 1e-10):
                    break

            return best_local_x, best_local_f, calls_used

        # --- Local Polisher: L-BFGS-B ---
        def local_polish_l_bfgs(x0, f0, max_calls):
            """
            Run L-BFGS-B from x0. Returns best_x, best_f, calls_used.
            """
            nonlocal n_calls, best_x, best_f

            calls_used = 0
            x_best = x0.copy()
            f_best = f0

            def objective(x):
                nonlocal calls_used
                if calls_used >= max_calls or n_calls >= budget or check_time():
                    return f_best  # Return current best to stop optimization
                val, x_eval = call_func(x)
                if val is None:
                    return f_best
                calls_used += 1
                return val

            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_calls, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < f_best:
                    x_best = np.clip(result.x, -5, 5)
                    f_best = result.fun
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()
            except Exception:
                pass

            return x_best, f_best, calls_used

        # --- Global Search Strategy with Adaptive Restarts ---
        # Phase 1: Initial exploration with CMA-ES from a random start
        # Phase 2: Adaptive restarts if stagnation detected
        # Phase 3: Local refinement with L-BFGS-B

        # Initial random sample to get a starting point
        x_init = rng.uniform(-5, 5, dim)
        f_init, x_init_eval = call_func(x_init)
        if f_init is None:
            return best_x, best_f
        update_best(x_init_eval, f_init)

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
        pop_size = max(pop_size, 6)
        sigma_init = 2.0  # Initial step size

        # Run CMA-ES with adaptive restarts
        restarts = 0
        max_restarts = 5
        stagnation_threshold = 50  # Calls without improvement before restart
        last_improvement_call = n_calls
        best_f_before_restart = best_f

        while n_calls < budget and not check_time() and restarts < max_restarts:
            # Run CMA-ES
            max_cma_calls = min(budget - n_calls, 100 + restarts * 20)
            x_result, f_result, calls_used = cma_es(best_x, sigma_init, pop_size, max_cma_calls)

            # Check for improvement
            if f_result < best_f_before_restart - 1e-10:
                # Improved, reset stagnation counter
                last_improvement_call = n_calls
                best_f_before_restart = best_f
            else:
                # Check for stagnation
                if n_calls - last_improvement_call > stagnation_threshold:
                    restarts += 1
                    # Restart from a new random point or a perturbation of best
                    if restarts % 2 == 1:
                        # Random restart
                        x_new = rng.uniform(-5, 5, dim)
                    else:
                        # Perturbation of best
                        x_new = best_x + 1.0 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)

                    f_new, x_new_eval = call_func(x_new)
                    if f_new is not None:
                        update_best(x_new_eval, f_new)
                        last_improvement_call = n_calls
                        best_f_before_restart = best_f
                        sigma_init = max(1.0, sigma_init * 0.5)  # Shrink step size for restarts
                    continue

            # If we used most of the budget, break
            if n_calls >= budget * 0.8:
                break

        # Phase 3: Local refinement with L-BFGS-B on the best point
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            max_l_bfgs_calls = min(remaining, 50)
            if max_l_bfgs_calls > 5:
                x_polished, f_polished, calls_used = local_polish_l_bfgs(best_x, best_f, max_l_bfgs_calls)

        # Final check
        if best_x is None:
            return None

        return best_x, best_f

# EVOLVE-BLOCK-END
