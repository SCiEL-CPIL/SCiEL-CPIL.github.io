"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-model multi-objective optimizer with weight-based mutation intensities."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Define island weights: uniform spread over the simplex
        n_islands = 6
        weights = []
        for i in range(n_islands):
            w = np.ones(n_obj)
            # Create diverse weight vectors
            if i == 0:
                w = np.array([0.7, 0.15, 0.15])
            elif i == 1:
                w = np.array([0.15, 0.7, 0.15])
            elif i == 2:
                w = np.array([0.15, 0.15, 0.7])
            elif i == 3:
                w = np.array([0.4, 0.4, 0.2])
            elif i == 4:
                w = np.array([0.4, 0.2, 0.4])
            else:
                w = np.array([0.2, 0.4, 0.4])
            w = w / w.sum()
            weights.append(w)

        # Island population size
        pop_per_island = max(5, min(15, budget // (n_islands * 3)))
        total_init = pop_per_island * n_islands
        if total_init > budget:
            pop_per_island = max(3, budget // n_islands)
            total_init = pop_per_island * n_islands

        # Initialize all islands
        islands_X = []
        islands_F = []
        used = 0
        for _ in range(n_islands):
            X_i = lo + (hi - lo) * rng.random((pop_per_island, dim))
            F_i = np.array([problem(x) for x in X_i])
            used += pop_per_island
            islands_X.append(X_i)
            islands_F.append(F_i)

        # Track global best non-dominated set
        global_X = np.vstack(islands_X)
        global_F = np.vstack(islands_F)

        def hypervolume_contribution(F_set, ref_point):
            """Compute hypervolume contribution of non-dominated points."""
            # Filter points better than reference in all objectives
            mask = np.all(F_set < ref_point, axis=1)
            F_set = F_set[mask]
            if len(F_set) == 0:
                return 0.0
            if len(F_set) == 1:
                return np.prod(ref_point - F_set[0])
            # Simple O(n^3) hypervolume for small sets
            # Sort by first objective
            idx = np.argsort(F_set[:, 0])
            F_sorted = F_set[idx]
            hv = 0.0
            n = len(F_sorted)
            for i in range(n):
                # Hypervolume contribution of point i
                # Points that dominate point i in objectives 1..n_obj-1
                if i == 0:
                    width_0 = ref_point[0] - F_sorted[i, 0]
                else:
                    width_0 = F_sorted[i-1, 0] - F_sorted[i, 0]
                if width_0 <= 0:
                    continue
                # For remaining objectives, find min among points 0..i
                min_vals = np.min(F_sorted[:i+1, 1:], axis=0)
                width_rest = np.prod(ref_point[1:] - min_vals)
                hv += width_0 * width_rest
            return hv

        ref_point = np.array([3.0, 5.0, 7.0])

        while used < budget:
            if time.time() > deadline:
                break

            # Select island to evolve (round-robin or random)
            island_idx = int(used) % n_islands
            X_i = islands_X[island_idx]
            F_i = islands_F[island_idx]
            w = weights[island_idx]

            # Compute weighted scalarization for parent selection
            scores = F_i @ w
            # Select top parents
            n_parents = max(2, len(X_i) // 2)
            parent_idx = np.argsort(scores)[:n_parents]
            parent = X_i[parent_idx[rng.integers(n_parents)]]

            # Weight-specific mutation intensity
            # Islands with higher weight on a particular objective get larger mutation in that direction
            base_intensity = 0.15
            # Scale mutation by inverse weight (more mutation where weight is lower to explore)
            mutation_scale = base_intensity / (w + 1e-8)
            mutation_scale = np.clip(mutation_scale, 0.05, 0.5)
            
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst individual in this island based on weighted score
            child_score = fc @ w
            worst_idx = int(np.argmax(scores))
            X_i[worst_idx] = child
            F_i[worst_idx] = fc
            islands_X[island_idx] = X_i
            islands_F[island_idx] = F_i

            # Update global set
            global_X = np.vstack(islands_X)
            global_F = np.vstack(islands_F)

        return global_X
# EVOLVE-BLOCK-END
