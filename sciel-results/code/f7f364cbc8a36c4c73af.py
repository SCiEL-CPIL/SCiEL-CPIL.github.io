"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: mix of random and boundary points
        pop_size = min(20, self.budget // 5)
        X = np.empty((pop_size, dim))
        n_half = pop_size // 2
        X[:n_half] = lo + span * rng.random((n_half, dim))
        for i in range(n_half, pop_size):
            corners = rng.random(dim) < 0.5
            X[i] = np.where(corners, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True

            non_dom_idx = np.where(~dominated)[0]
            
            # Select parent from non-dominated set
            if len(non_dom_idx) > 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                parent_idx = rng.integers(n)

            # Adaptive mutation: step size decreases over time
            progress = used / self.budget
            base_step = 0.15 * (1.0 - 0.7 * progress)
            
            r = rng.random()
            if r < 0.2 and len(F) >= 2 and len(non_dom_idx) > 1:
                # Crossover with another non-dominated point
                other_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                alpha = rng.random()
                child = alpha * X[parent_idx] + (1 - alpha) * X[other_idx]
                child += 0.05 * span * rng.standard_normal(dim)
            elif r < 0.6:
                # Gaussian mutation
                child = X[parent_idx] + base_step * span * rng.standard_normal(dim)
            else:
                # Larger jump to explore
                child = X[parent_idx] + 0.3 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point: prefer dominated points, then worst sum
            worst_score = np.sum(F, axis=1)
            for i in range(n):
                if dominated[i]:
                    worst_score[i] += 1e10
            worst_idx = int(np.argmax(worst_score))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
