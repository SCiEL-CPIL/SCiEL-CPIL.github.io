"""Differential Evolution with adaptive parameters (jDE) for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameter control for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lambda_pop = 20
        n = dim

        # Initialize population uniformly in [-5, 5]^n
        pop = rng.uniform(-5, 5, (lambda_pop, n))
        pop_f = np.array([func(x) for x in pop])
        total_calls = lambda_pop

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # jDE parameters
        F = 0.5  # Scaling factor
        CR = 0.9  # Crossover rate

        # History for adaptive control
        F_history = []
        CR_history = []
        success_count = 0
        trial_count = 0

        # Main loop
        while total_calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring for each individual
            new_pop = []
            new_pop_f = []
            
            for i in range(lambda_pop):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break

                # jDE: adapt F and CR based on success history
                if len(F_history) > 10:
                    # Use mean of successful parameters
                    F = np.mean([F_history[i][0] for i in range(len(F_history)) if F_history[i][1]])
                    CR = np.mean([F_history[i][1] for i in range(len(F_history)) if F_history[i][1]])
                else:
                    F = 0.5
                    CR = 0.9

                # Add randomness to F and CR
                F_i = max(0.1, min(1.0, F + rng.normal(0, 0.1)))
                CR_i = max(0.1, min(0.95, CR + rng.normal(0, 0.1)))

                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(lambda_pop, 3, replace=False)
                while r1 == i:
                    r1 = np.random.choice(lambda_pop)
                while r2 == i or r2 == r1:
                    r2 = np.random.choice(lambda_pop)
                while r3 == i or r3 in [r1, r2]:
                    r3 = np.random.choice(lambda_pop)

                # DE/rand/1 mutation
                v = pop[r1] + F_i * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                j_rand = rng.integers(n)
                u = rng.uniform(size=n)
                u[j_rand] = 1  # Ensure at least one gene is taken from v
                trial = np.where(u <= CR_i, v, pop[i])
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                trial_f = func(trial)
                total_calls += 1

                # Selection
                if trial_f <= pop_f[i]:
                    new_pop.append(trial)
                    new_pop_f.append(trial_f)
                    success_count += 1
                    F_history.append((F_i, CR_i, True))
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])
                    F_history.append((F_i, CR_i, False))

                trial_count += 1

            # Update population
            pop = np.array(new_pop)
            pop_f = np.array(new_pop_f)

            # Update best
            current_best_idx = np.argmin(pop_f)
            if pop_f[current_best_idx] < best_f:
                best_f = pop_f[current_best_idx]
                best_x = pop[current_best_idx].copy()

            # Prune history to avoid memory growth
            if len(F_history) > 1000:
                F_history = F_history[-500:]

            # Check for convergence: if all individuals are very close
            if np.std(pop_f) < 1e-10:
                break

        # Local polish around best point
        if best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            for _ in range(min(50, budget - total_calls)):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    best_f = fx
                    best_x = x.copy()
                else:
                    sigma *= 0.8

                if sigma < 1e-7:
                    sigma = 0.02

        return best_x, best_f
# EVOLVE-BLOCK-END
