"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + (1+1)-ES with coordinate-descent polish for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with population-based search
        # Use multiple random starts to find good basins
        n_starts = min(5, budget // 100)
        best_x = np.zeros(dim)
        best_fx = func(best_x)
        total_calls = 1

        # Try a few random starting points
        for i in range(n_starts):
            if total_calls >= budget or time.time() >= deadline - 50:
                break
            x_init = np.random.uniform(-4, 4, dim)
            fx_init = func(x_init)
            total_calls += 1
            if fx_init < best_fx:
                best_fx = fx_init
                best_x = x_init.copy()

        # Phase 2: CMA-ES-like population search
        pop_size = min(15, max(5, budget // 40))
        x_mean = best_x.copy()
        sigma = 2.0
        C = np.eye(dim)  # covariance matrix
        lambda_ = pop_size

        # Track best
        best_fx_cma = best_fx
        best_x_cma = best_x.copy()

        # CMA-ES loop
        cma_budget = budget // 2
        cma_calls = 0
        stagnation = 0

        while cma_calls < cma_budget and total_calls < budget and time.time() < deadline - 100:
            if time.time() >= deadline - 100:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                if total_calls >= budget or time.time() >= deadline - 100:
                    break
                # Sample from multivariate normal
                z = np.random.randn(dim)
                # Transform using C (simplified: use diagonal approximation)
                y = x_mean + sigma * np.diag(np.sqrt(np.diag(C))) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                cma_calls += 1
                population.append((y, fy))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda x: x[1])

            # Update best
            if population[0][1] < best_fx_cma:
                best_fx_cma = population[0][1]
                best_x_cma = population[0][0].copy()
                if best_fx_cma < best_fx:
                    best_fx = best_fx_cma
                    best_x = best_x_cma.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            # Update mean (weighted average of best half)
            mu = max(1, lambda_ // 2)
            weights = np.ones(mu) / mu
            x_mean = sum(weights[i] * population[i][0] for i in range(mu))
            x_mean = np.clip(x_mean, -5, 5)

            # Update sigma (1/5 success rule approximation)
            success_rate = sum(1 for _, fy in population[:mu] if fy < population[0][1] + 1e-10) / mu
            if success_rate > 0.2:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = max(1e-3, min(5.0, sigma))

            # Update covariance (simplified: track variance in each dimension)
            for d in range(dim):
                var_d = np.var([p[0][d] for p in population[:mu]])
                C[d, d] = max(1e-6, var_d + 1e-6)

            # If stagnating, try a restart from a random point near best
            if stagnation > 30:
                # Perturb mean slightly
                x_mean = best_x_cma + np.random.randn(dim) * 0.5
                x_mean = np.clip(x_mean, -5, 5)
                sigma = max(sigma * 1.2, 0.5)
                stagnation = 0

        # Phase 3: (1+1)-ES fine-tuning
        x = best_x.copy()
        fx = best_fx
        sigma_es = max(sigma * 0.3, 0.1)

        es_budget = budget - total_calls
        es_calls = 0
        es_stagnation = 0

        while es_calls < es_budget and total_calls < budget and time.time() < deadline - 50:
            if time.time() >= deadline - 50:
                break

            y = np.clip(x + sigma_es * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1
            es_calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    es_stagnation = 0
            else:
                es_stagnation += 1

            # Adapt step size
            if es_stagnation % 10 == 0:
                if es_stagnation < 20:
                    sigma_es *= 0.9
                else:
                    sigma_es *= 0.8
            elif fy < fx:
                sigma_es *= 1.02

            sigma_es = max(1e-4, min(2.0, sigma_es))

            # If heavily stagnated, try coordinate descent
            if es_stagnation > 50 and time.time() < deadline - 30:
                break

        # Phase 4: Coordinate descent polish (exploits Schwefel separability)
        if total_calls < budget and time.time() < deadline - 10:
            cd_x = best_x.copy()
            cd_fx = best_fx
            cd_step = max(sigma_es * 0.5, 0.05)
            cd_iterations = min(20, (budget - total_calls) // (2 * dim))

            for it in range(cd_iterations):
                if time.time() >= deadline - 10:
                    break
                for d in range(dim):
                    if total_calls >= budget or time.time() >= deadline - 10:
                        break
                    # Try positive direction
                    trial_pos = cd_x.copy()
                    trial_pos[d] += cd_step
                    trial_pos = np.clip(trial_pos, -5, 5)
                    fpos = func(trial_pos)
                    total_calls += 1

                    if fpos < cd_fx:
                        cd_x = trial_pos
                        cd_fx = fpos
                        if cd_fx < best_fx:
                            best_fx = cd_fx
                            best_x = cd_x.copy()
                    else:
                        # Try negative direction
                        trial_neg = cd_x.copy()
                        trial_neg[d] -= cd_step
                        trial_neg = np.clip(trial_neg, -5, 5)
                        fneg = func(trial_neg)
                        total_calls += 1

                        if fneg < cd_fx:
                            cd_x = trial_neg
                            cd_fx = fneg
                            if cd_fx < best_fx:
                                best_fx = cd_fx
                                best_x = cd_x.copy()

                # Shrink step size
                cd_step *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
