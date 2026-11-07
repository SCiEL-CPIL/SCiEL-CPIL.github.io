# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a population-based approach with
    DTLZ4-aware initialization and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dtlz4_front_sample(self, rng, n):
        """Sample points near the DTLZ4 Pareto front.
        For DTLZ4 with M=3 objectives, x[0:2] control the trade-off and x[2:] should be small.
        Actually for DTLZ4: x[0:M-1] = x[0:2] are the position variables, x[2:] are the distance variables.
        The Pareto front is f1 = x0, f2 = x1*(1-x0), f3 = (1-x1)*(1-x0).
        So to get good points: x[0] in [0,1], x[1] in [0,1], x[2:] small (near 0 or 1 depending on alpha).
        DTLZ4 has alpha=100, so the transformation x_i' = x_i^(1/alpha) means small x_i map to even smaller values.
        Wait, DTLZ4: x_i' = x_i^(1/alpha) for i >= M-1. With alpha=100, x_i^(1/100) is close to 1 for most x_i.
        So to minimize g, we want x_i close to 0 or 1? No, g = sum(x_i' - 0.5)^2 for i >= M-1.
        So x_i' should be 0.5, meaning x_i = 0.5^alpha = 0.5^100 which is extremely small.
        So x[2:] should be very small (near 0) to get g near 0.
        """
        points = np.zeros((n, self.dim))
        # x[0] and x[1] are the position variables
        points[:, 0] = rng.random(n)
        points[:, 1] = rng.random(n)
        # x[2:] should be very small so that x_i^(1/100) ≈ 0.5
        # x_i = 0.5^100 ≈ 7.9e-31, so just use very small values
        points[:, 2:] = rng.random((n, self.dim - 2)) * 1e-6
        return points

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (minimization)."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(i + 1, n):
                if not nd[j]:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    nd[j] = False
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
        return np.where(nd)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Initialize population: half from DTLZ4-aware front, half random
        pop_size = min(40, budget // 5)
        n_front = pop_size // 2
        n_rand = pop_size - n_front

        X = np.zeros((pop_size, dim))
        X[:n_front] = self._dtlz4_front_sample(rng, n_front)
        X[n_front:] = lo + (hi - lo) * rng.random((n_rand, dim))
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main search loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_idx = self._non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]

            # Generate multiple children per iteration to make progress faster
            n_children = min(5, budget - used)
            for _ in range(n_children):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parent via tournament
                p = parents[rng.integers(len(parents))]

                # Mutate: mix of small and large steps
                if rng.random() < 0.3:
                    # Large step
                    step = 0.3 * (hi - lo) * rng.standard_normal(dim)
                else:
                    # Small step
                    step = 0.05 * (hi - lo) * rng.standard_normal(dim)

                child = p + step
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst point (by sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
# EVOLVE-BLOCK-END
