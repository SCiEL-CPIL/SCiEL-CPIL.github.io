"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using a hybrid strategy:
    1. Sobol quasi-random initialization for broad coverage.
    2. NSGA-II style evolutionary search with tournament selection
       and crowding distance for diversity.
    3. Deadline guard to ensure completion within wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Sobol quasi-random initial population
        pop_size = min(40, budget // 3)
        # Use a simple quasi-random sequence (Halton-like via low-discrepancy)
        # We'll use scipy if available, otherwise a simple LCG-based quasi-random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            init_points = sampler.random(pop_size)
        except ImportError:
            # Fallback: use numpy random with stratification
            init_points = np.random.uniform(0, 1, (pop_size, dim))

        X = lo + init_points * span
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Check deadline
        if time.time() > deadline - 10:
            return X

        # Helper: compute non-dominated front (rank 0)
        def get_non_dominated(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        nd[j] = False
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Helper: crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-12:
                    continue
                order = np.argsort(f_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (f_max - f_min)
            return cd

        # Helper: tournament selection
        def tournament_select(F, k=2):
            n = len(F)
            # Get non-dominated
            nd_idx = get_non_dominated(F)
            if len(nd_idx) >= k:
                # Pick from non-dominated
                idx = nd_idx[np.random.randint(len(nd_idx))]
                return idx
            # If not enough, pick from all
            return np.random.randint(n)

        # Main evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Get non-dominated front
            nd_idx = get_non_dominated(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Compute crowding distance on non-dominated
            cd = crowding_distance(F, nd_idx)

            # Select parents
            parent1_idx = nd_idx[np.random.randint(len(nd_idx))]
            parent2_idx = nd_idx[np.random.randint(len(nd_idx))]

            # Crossover: blend
            alpha = np.random.uniform(0.2, 0.8)
            child = alpha * X[parent1_idx] + (1 - alpha) * X[parent2_idx]

            # Mutation: Gaussian
            sigma = 0.05 * span
            child = child + sigma * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            # Use a simple heuristic: replace the point with the largest
            # crowding distance among dominated points, or the worst non-dominated
            dominated = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = False
                        break

            if np.any(dominated):
                # Replace a dominated point
                dom_idx = np.where(dominated)[0]
                replace_idx = dom_idx[np.random.randint(len(dom_idx))]
            else:
                # Replace the non-dominated point with largest crowding distance
                replace_idx = nd_idx[np.argmax(cd)]

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
