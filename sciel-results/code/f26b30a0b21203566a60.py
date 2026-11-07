import numpy as np
import time


class Optimizer:
    """MOEA/D style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(50, max(10, self.budget // 4))
        remaining = self.budget

        # Generate weight vectors for MOEA/D
        # Using random normalized weights for simplicity and coverage
        weights = rng.random((pop_size, self.n_obj))
        weights /= weights.sum(axis=1, keepdims=True)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Track best and worst objective values for normalization
        best_F = F.min(axis=0)
        worst_F = F.max(axis=0)

        def tchebycheff_fitness(F, weights, best_F, worst_F):
            # Normalize objectives
            range_F = worst_F - best_F
            range_F[range_F < 1e-10] = 1.0
            F_norm = (F - best_F) / range_F

            # Calculate Tchebycheff distance
            # D = max_i (w_i * F_norm_i)
            prod = F_norm * weights
            return prod.max(axis=1)

        # Calculate initial fitness
        fitness = tchebycheff_fitness(F, weights, best_F, worst_F)

        while remaining > 0:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            n_off = min(pop_size, remaining)
            off_X = np.empty((n_off, self.dim))

            for i in range(n_off):
                # Select parents
                p = rng.integers(pop_size)
                q = rng.integers(pop_size)

                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    eta = 20
                    alpha = rng.random()
                    if alpha <= 0.5:
                        beta = (2 * alpha) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - alpha)) ** (1 / (eta + 1))
                    beta = 1.0 / beta
                    child = 0.5 * ((1 + beta) * X[p] + (1 - beta) * X[q])
                else:
                    child = X[p].copy()

                # Polynomial mutation
                eta_m = 20
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        delta = (2 * rng.random()) ** (1 / (eta_m + 1)) - 1
                        span = hi[d] - lo[d]
                        child[d] += delta * span

                child = np.clip(child, lo, hi)
                off_X[i] = child

            # Evaluate offspring
            off_F = np.array([problem(x) for x in off_X])
            remaining -= n_off

            # Update best and worst for normalization
            F_all = np.vstack([F, off_F])
            new_best_F = F_all.min(axis=0)
            new_worst_F = F_all.max(axis=0)

            # Calculate fitness for offspring using updated bounds
            off_fitness = tchebycheff_fitness(off_F, weights, new_best_F, new_worst_F)

            # Recalculate parent fitness with new bounds for fair comparison
            parent_fitness_new = tchebycheff_fitness(F, weights, new_best_F, new_worst_F)

            # Replace parents if offspring are better in their respective subproblems
            for i in range(n_off):
                if off_fitness[i] < parent_fitness_new[i]:
                    X[i] = off_X[i]
                    F[i] = off_F[i]
                    fitness[i] = off_fitness[i]

            # Update global best and worst for next iteration
            best_F = new_best_F
            worst_F = new_worst_F

        return X
