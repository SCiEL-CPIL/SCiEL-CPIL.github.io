"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard, adaptive mutation, and non-dominated replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the seeded np.random (evaluator seeds it before calling)
        # Initial population: LHS-like spread for diversity
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for i in range(pop_size):
            X[i] = lo + span * (np.arange(dim) / (dim - 1 + 1e-12) * 0 + np.random.rand(dim))
            F[i] = np.array(problem(X[i]))

        used = pop_size

        # Track best per objective for adaptive mutation scaling
        best_obj = F.min(axis=0)
        worst_obj = F.max(axis=0)

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                if dominated[i]:
                    continue
                for j in range(pop_size):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True

            pareto_idx = np.where(~dominated)[0]
            if len(pareto_idx) == 0:
                pareto_idx = np.arange(pop_size)

            # Select two parents from Pareto set (or whole pop if needed)
            if len(pareto_idx) >= 2:
                p1_idx, p2_idx = np.random.choice(pareto_idx, size=2, replace=False)
            else:
                p1_idx, p2_idx = np.random.choice(pop_size, size=2, replace=True)

            # Crossover: uniform blend
            mask = np.random.rand(dim) > 0.5
            child = np.where(mask, X[p1_idx], X[p2_idx])

            # Adaptive mutation: step size decreases as budget depletes
            progress = used / budget
            sigma = 0.15 * (1.0 - 0.7 * progress)
            child = child + sigma * span * np.random.randn(dim)
            child = np.clip(child, lo, hi)

            fc = np.array(problem(child))
            used += 1

            # Update best/worst per objective
            best_obj = np.minimum(best_obj, fc)
            worst_obj = np.maximum(worst_obj, fc)

            # Replacement: if child is non-dominated, replace worst dominated or worst overall
            # Check if child dominates any existing
            child_dominates = np.zeros(pop_size, dtype=bool)
            dominated_by_child = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    child_dominates[i] = True
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    dominated_by_child[i] = True

            if np.any(child_dominates):
                # Replace one of the dominated solutions
                dom_targets = np.where(child_dominates)[0]
                replace_idx = dom_targets[np.argmax(F[dom_targets].sum(axis=1))]
            elif not np.any(dominated_by_child):
                # Child is non-dominated; replace the worst solution (highest sum)
                replace_idx = int(np.argmax(F.sum(axis=1)))
            else:
                # Child is dominated; only replace if it's better than the worst in some objective
                worst_idx = int(np.argmax(F.sum(axis=1)))
                if np.any(fc < F[worst_idx]):
                    replace_idx = worst_idx
                else:
                    continue  # skip this child

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
