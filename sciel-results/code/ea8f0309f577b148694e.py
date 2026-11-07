"""Rewritten optimizer for ZDT4 with deadline guard and stronger search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using a simple EA with crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        # ZDT4 specific: x1 is in [-5, 5], others in [0, 1]
        # The Pareto front is achieved when x1=0 and each other xi is either
        # near 0 or near 1 (the sin term in ZDT4 creates many local optima)

        # Strategy: use a population-based search with:
        # 1. Initialize with points that have x1 near 0 and xi near 0 or 1
        # 2. Use mutation that can jump between the binary-like states
        # 3. Use crowding distance for diversity

        pop_size = min(50, budget // 4)
        if pop_size < 10:
            pop_size = 10

        def make_individual():
            x = np.zeros(n)
            x[0] = rng.uniform(-0.5, 0.5)  # x1 near 0 for Pareto front
            for i in range(1, n):
                # Mix of near 0 and near 1 to cover the front
                if rng.random() < 0.5:
                    x[i] = rng.uniform(0, 0.1)
                else:
                    x[i] = rng.uniform(0.9, 1.0)
            return np.clip(x, lo, hi)

        # Initialize population
        X = np.array([make_individual() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return non-dominated set indices."""
            n_pts = len(F)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def crowding_distance(F, nd_mask):
            """Compute crowding distance for non-dominated points."""
            cd = np.zeros(len(F))
            idx = np.where(nd_mask)[0]
            if len(idx) <= 2:
                cd[idx] = np.inf
                return cd
            for m in range(self.n_obj):
                sorted_idx = idx[np.argsort(F[idx, m])]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
                if f_range > 1e-15:
                    for k in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[k]] += (
                            F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]
                        ) / f_range
            return cd

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = fast_non_dominated_sort(F)
            nd_count = np.sum(nd_mask)

            # Select parents from non-dominated set
            if nd_count > 0:
                parents_idx = np.where(nd_mask)[0]
                # Use crowding distance to select diverse parents
                cd = crowding_distance(F, nd_mask)
                # Select top half by crowding distance
                n_select = max(2, len(parents_idx) // 2)
                sorted_by_cd = parents_idx[np.argsort(cd[parents_idx])[::-1]]
                parent_pool = sorted_by_cd[:n_select]
            else:
                parent_pool = np.arange(len(F))

            # Generate child via mutation
            parent = X[parent_pool[rng.integers(len(parent_pool))]]
            # Mutation: small perturbation
            mutation_scale = 0.05 * (hi - lo)
            child = parent + rng.standard_normal(n) * mutation_scale
            child = np.clip(child, lo, hi)

            # Additional strategy: sometimes do a larger jump for x1 or xi
            if rng.random() < 0.1:
                # Large jump for x1
                child[0] = rng.uniform(-0.5, 0.5)
            if rng.random() < 0.1:
                # Flip a random xi between near 0 and near 1
                i = rng.integers(1, n)
                if child[i] < 0.5:
                    child[i] = rng.uniform(0.9, 1.0)
                else:
                    child[i] = rng.uniform(0, 0.1)

            fc = problem(child)
            used += 1

            # Replace worst individual (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
