"""Multi-objective DE optimizer for WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using DE/rand/1/bin with non-dominated archive for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: keep it reasonable for 200 evaluations
        pop_size = 20
        # DE parameters
        CR = 0.7
        F = 0.5

        # Initialize population uniformly
        rng = np.random.RandomState(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F_obj = np.array([problem(x) for x in X])
        used = pop_size

        # Track best archive (non-dominated set)
        # We'll maintain the full population and use non-dominated sorting for selection

        def dominates(a, b):
            """True if a dominates b (a <= b in all objectives and a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_arr):
            """Return indices of non-dominated solutions."""
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(i + 1, n):
                    if not nd[j]:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        nd[j] = False
                    elif dominates(F_arr[j], F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Main DE loop
        while used < budget:
            # Check time
            if time.time() > deadline - 2:
                break

            # Select non-dominated parents for diversity
            nd_idx = non_dominated_sort(F_obj)
            if len(nd_idx) < 3:
                # Fall back to all if too few non-dominated
                parent_pool = X
                parent_idx = np.arange(pop_size)
            else:
                parent_pool = X[nd_idx]
                parent_idx = nd_idx

            n_parents = len(parent_pool)
            if n_parents < 3:
                # Can't do DE with fewer than 3 distinct parents, use mutation
                parent = parent_pool[rng.randint(n_parents)]
                child = parent + 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
            else:
                # DE/rand/1/bin: pick 3 distinct parents
                p1, p2, p3 = rng.choice(n_parents, size=3, replace=False)
                v = parent_pool[p1] + F * (parent_pool[p2] - parent_pool[p3])

                # Binary crossover
                crossover_mask = rng.random(dim) < CR
                # Ensure at least one gene is from v
                if not np.any(crossover_mask):
                    j_rand = rng.randint(dim)
                    crossover_mask[j_rand] = True

                parent = parent_pool[rng.randint(n_parents)]
                child = np.where(crossover_mask, v, parent)
                child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst solution (by sum of objectives, simple heuristic)
            # or use tournament selection
            worst_idx = np.argmax(np.sum(F_obj, axis=1))
            X[worst_idx] = child
            F_obj[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
