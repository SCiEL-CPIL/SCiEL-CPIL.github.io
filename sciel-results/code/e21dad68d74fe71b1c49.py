"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive sigma for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0

        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # (mu+lambda)-ES parameters
        mu = 4
        lam = 16
        sigma = 1.5
        sigma_min = 1e-10
        sigma_max = 5.0
        c_sigma = 1.0 / (4 * dim + 4)  # learning rate for sigma
        stagnation_limit = 50

        # Initial population: spread out in the box
        population = []
        for i in range(lam):
            if calls >= budget or time.time() - start_time > max_time:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            population.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        population.sort(key=lambda t: t[0])

        # (mu+lambda)-ES main loop
        iters_no_improve = 0
        while calls < budget and time.time() - start_time < max_time:
            # Select mu best from population
            parents = [x for _, x in population[:mu]]
            
            # Compute mean of parents
            mean_x = np.mean(parents, axis=0)
            
            # Generate lambda offspring
            offspring = []
            for j in range(lam):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                direction = rng.standard_normal(dim)
                candidate = mean_x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                offspring.append((fc, candidate.copy()))
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()

            if not offspring:
                break

            # Combine parents and offspring, select best lam
            combined = population + offspring
            combined.sort(key=lambda t: t[0])
            population = combined[:lam]

            # Check for improvement
            if population[0][0] < best_f + 1e-12:
                pass  # already tracked
            else:
                iters_no_improve += 1
            
            # Actually track improvement properly
            if population[0][0] < best_f - 1e-12:
                iters_no_improve = 0
            
            # Adapt sigma based on success rate
            # Success rate: fraction of offspring that are better than mean parent fitness
            parent_best = parents[0][0] if parents else best_f
            successes = sum(1 for f, _ in offspring if f < parent_best)
            success_rate = successes / len(offspring) if offspring else 0.0
            
            # 1/5 rule-like adaptation
            if success_rate > 0.2:
                sigma = min(sigma * 1.05, sigma_max)
            elif success_rate < 0.2:
                sigma = max(sigma * 0.95, sigma_min)

            # Stagnation restart
            if iters_no_improve >= stagnation_limit:
                # Reset population around current best with larger spread
                population = []
                for i in range(lam):
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    x = best_x + rng.normal(0, 2.0, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    population.append((fx, x.copy()))
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                population.sort(key=lambda t: t[0])
                sigma = 1.5
                iters_no_improve = 0

        # Final polish: fine-tuning around the best solution
        if calls < budget and time.time() - start_time < max_time:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(50):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                if fc < fx:
                    x = candidate
                    fx = fc
                    sigma = min(sigma * 1.02, 0.5)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.98, 1e-6)

        return best_x, best_f
# EVOLVE-BLOCK-END
