"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search for WFG4 with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = min(50, self.budget // 4)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def is_non_dominated(F):
            """Return boolean array: True if point is non-dominated."""
            n = len(F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom[i] = False
                        break
            return non_dom

        # Main evolutionary phase: use ~70% of budget
        evolve_budget = int(self.budget * 0.7)
        refine_budget = self.budget - evolve_budget

        mut_scale = 0.15

        while used < evolve_budget:
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = is_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)

            # Select parent from non-dominated set
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]

            # Mutation
            child = parent + mut_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual in population (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        # Local refinement phase: polish non-dominated points with small perturbations
        nd_mask = is_non_dominated(F)
        nd_indices = np.where(nd_mask)[0]

        if len(nd_indices) == 0:
            nd_indices = np.arange(pop_size)

        # Sort non-dominated points by sum (best first)
        nd_sums = np.sum(F[nd_indices], axis=1)
        nd_sorted = nd_indices[np.argsort(nd_sums)]

        # Refine each non-dominated point with small perturbations
        refine_scale = 0.02
        points_to_refine = nd_sorted[:min(15, len(nd_sorted))]

        for pi in points_to_refine:
            if used >= self.budget or time.time() > deadline:
                break

            # Try several small perturbations around this point
            for _ in range(3):
                if used >= self.budget or time.time() > deadline:
                    break

                child = X[pi] + refine_scale * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # If this improves (is non-dominated or better sum), add to population
                # Check if fc is non-dominated w.r.t. current F
                is_nd = True
                for fi in F:
                    if np.all(fi <= fc) and np.any(fi < fc):
                        is_nd = False
                        break

                if is_nd:
                    # Replace worst
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = child
                    F[worst] = fc

        return X
# EVOLVE-BLOCK-END
