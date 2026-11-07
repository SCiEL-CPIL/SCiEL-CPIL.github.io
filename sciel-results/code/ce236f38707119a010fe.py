"""Multi-objective optimizer for ZDT1 using a decomposition-based approach (NSGA-III style)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


class Optimizer:
    """Decomposition-based multi-objective optimizer (NSGA-III inspired) for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use global random state (seeded by evaluator)
        # We'll use np.random directly to respect the seed

        # Population size: keep it manageable
        pop_size = min(50, budget // 5)
        if pop_size < 5:
            pop_size = 5

        # Generate reference points using a simplex-lattice or weighted sum approach
        # For 2 objectives, we can use a set of weights
        n_weights = pop_size
        # Create weights that sum to 1
        # Use a grid of weights
        weights = []
        # For 2 objectives, weights are (w, 1-w)
        # Use evenly spaced weights
        for i in range(n_weights):
            w = i / max(1, n_weights - 1)
            weights.append(np.array([w, 1.0 - w]))
        weights = np.array(weights)

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Precompute reference point for normalization
        # We'll normalize objectives dynamically

        def normalize(f):
            """Normalize objectives to [0, 1] range."""
            f_min = np.min(F, axis=0)
            f_max = np.max(F, axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            return (f - f_min) / f_range

        def crowding_distance(f):
            """Compute crowding distance for each point."""
            n = len(f)
            cd = np.zeros(n)
            for m in range(n_obj):
                order = np.argsort(f[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_min = f[order[0], m]
                f_max = f[order[-1], m]
                if f_max - f_min > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (f[order[i + 1], m] - f[order[i - 1], m]) / (f_max - f_min)
            return cd

        def fast_non_dominated_sort(f):
            """Return non-dominated set indices."""
            n = len(f)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            # Non-dominated points
            nondominated = np.where(domination_count == 0)[0]
            return nondominated

        # Main loop
        while used < budget:
            # Find non-dominated points
            nondom = fast_non_dominated_sort(F)

            # Select parents from non-dominated set
            if len(nondom) == 0:
                parents_idx = np.arange(pop_size)
            else:
                parents_idx = nondom

            # For each weight, try to create a child
            # We'll cycle through weights
            for w_idx in range(len(weights)):
                if used >= budget:
                    break

                w = weights[w_idx]

                # Select parent: choose from non-dominated points
                # Use a tournament-like selection based on distance to weight direction
                # Compute projection of each non-dominated point onto the weight vector
                if len(parents_idx) > 0:
                    # Normalize F for projection
                    f_norm = normalize(F)
                    projections = f_norm[parents_idx] @ w
                    # Select the best (lowest projection, since we're minimizing)
                    best_local = np.argmin(projections)
                    parent_idx = parents_idx[best_local]
                else:
                    parent_idx = np.random.randint(pop_size)

                # Crossover and mutation
                # Simple mutation: Gaussian perturbation
                sigma = 0.1
                child = X[parent_idx].copy()
                mutation = np.random.standard_normal(dim) * sigma * (hi - lo)
                child += mutation
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1

                # Replace the worst point
                # Use a simple strategy: replace the point with the worst hypervolume contribution
                # Or replace the point farthest from the ideal point
                # For simplicity, replace the point with the highest sum of objectives
                worst_idx = np.argmax(np.sum(F, axis=1))

                # But we should be smarter: only replace if child is better or non-dominated
                # Check if child dominates the worst point
                if np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                    X[worst_idx] = child
                    F[worst_idx] = fc
                else:
                    # Check if worst point dominates child
                    if np.all(F[worst_idx] <= fc) and np.any(F[worst_idx] < fc):
                        # Don't replace
                        pass
                    else:
                        # Neither dominates; replace if child has better weight-based fitness
                        # Compute weight-based fitness
                        f_min = np.min(F, axis=0)
                        f_max = np.max(F, axis=0)
                        f_range = f_max - f_min
                        f_range[f_range == 0] = 1.0

                        child_norm = (fc - f_min) / f_range
                        worst_norm = (F[worst_idx] - f_min) / f_range

                        child_fit = child_norm @ w
                        worst_fit = worst_norm @ w

                        if child_fit < worst_fit:
                            X[worst_idx] = child
                            F[worst_idx] = fc

        return X

# EVOLVE-BLOCK-END
