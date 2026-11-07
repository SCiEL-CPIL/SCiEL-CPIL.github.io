"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive DE and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim

        pop_size = 20
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        while used < self.budget:
            if time.time() > deadline:
                break
            nd = non_dominated_mask(F)
            front = X[nd]
            front_F = F[nd]
            n_front = len(front)

            if n_front >= 3:
                # DE crossover
                a = rng.integers(n_front)
                b = rng.integers(n_front)
                c = rng.integers(n_front)
                while b == a:
                    b = rng.integers(n_front)
                while c == a or c == b:
                    c = rng.integers(n_front)
                F1 = front[a]
                F2 = front[b]
                F3 = front[c]
                F4 = front[rng.integers(n_front)]
                CR = 0.9
                jrand = rng.integers(dim)
                v = F1 + 0.7 * (F2 - F3)
                for j in range(dim):
                    if rng.random() <= CR or j == jrand:
                        v[j] = v[j]
                child = v
            else:
                # Fallback: mutate a random individual
                idx = rng.integers(pop_size)
                child = X[idx] + 0.1 * span * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace the worst individual in the population
            # Use a simple crowding-like distance or just sum
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
