"""Multi-objective differential evolution optimizer for WFG7."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective DE with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin before 300s
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Population size: use a good fraction of budget
        N = int(min(40, max(10, budget // 5)))
        F_de = 0.8  # differential evolution scaling factor
        CR = 0.5    # crossover rate

        # Initialize population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        def dominates(a, b):
            """True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            """Return indices of non-dominated points in F (vectorized)."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                # Check if any other point dominates i
                # F[j] <= F[i] for all obj, F[j] < F[i] for at least one
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                dom = le & lt
                dom[i] = False
                if np.any(dom):
                    nd[i] = False
            return np.where(nd)[0]

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            n = len(nd_idx)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
            # Sort by objective m
                order = np.argsort(F[nd_idx, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin = F[nd_idx[order[0]], m]
                fmax = F[nd_idx[order[-1]], m]
                rng_m = fmax - fmin
                if rng_m < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (F[nd_idx[order[k+1]], m] - F[nd_idx[order[k-1]], m]) / rng_m
            return dist

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.perf_counter() > deadline:
                break

            # Find non-dominated set
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) < 2:
                nd_idx = np.arange(N)

            # Differential evolution: for each individual, create trial
            for i in range(N):
                if used >= budget:
                    break
                if time.perf_counter() > deadline:
                    break

                # Select three distinct random parents different from i
                candidates = [j for j in range(N) if j != i]
                if len(candidates) < 3:
                    continue
                p1, p2, p3 = rng.choice(candidates, size=3, replace=False)

                # Differential evolution mutation
                v = X[p1] + F_de * (X[p2] - X[p3])

                # Binomial crossover
                r = rng.random(dim)
                jrand = rng.integers(dim)
                trial = np.where(r < CR, v, X[i])
                trial[jrand] = v[jrand]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Selection: replace i if trial dominates i, or is non-dominated and has better crowding
                if dominates(f_trial, F[i]):
                    X[i] = trial
                    F[i] = f_trial
                elif not dominates(F[i], f_trial):
                    # Both non-dominated relative to each other: keep the one with better position
                    # Simple heuristic: keep the one with smaller sum (closer to origin)
                    if np.sum(f_trial) < np.sum(F[i]):
                        X[i] = trial
                        F[i] = f_trial

            # Periodic re-injection to maintain diversity
            if used < budget and N > 5:
                # Replace the worst point (by sum) with a random one every few generations
                if (used - N) % N == 0:
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = lo + span * rng.random(dim)
                    F[worst] = problem(X[worst])
                    used += 1

        return X
# EVOLVE-BLOCK-END
