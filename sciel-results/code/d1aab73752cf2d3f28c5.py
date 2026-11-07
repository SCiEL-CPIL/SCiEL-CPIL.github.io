"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with adaptive mutation and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Parameters
        pop_size = min(20, self.budget // 2)
        # Ensure at least 2
        pop_size = max(2, pop_size)

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track best known points for hypervolume
        # We'll maintain an archive of non-dominated points

        def non_dominated_mask(F):
            """Return boolean mask: True if point is non-dominated."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        def crowding_distances(F, nd_mask):
            """Compute crowding distances for non-dominated points."""
            n = len(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for k in range(self.n_obj):
                # Sort by objective k
                order = nd_idx[np.argsort(F[nd_idx, k])]
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_min = F[order[0], k]
                f_max = F[order[-1], k]
                if f_max - f_min < 1e-15:
                    continue
                for i in range(1, len(order) - 1):
                    dist[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / (f_max - f_min)
            return dist

        # Main evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = non_dominated_mask(F)
            nd_count = np.sum(nd_mask)

            # Selection: pick parents from non-dominated set
            if nd_count > 0:
                nd_idx = np.where(nd_mask)[0]
                # Use crowding distance for diversity
                cd = crowding_distances(F, nd_mask)
                # Select parent with highest crowding distance among non-dominated
                # Weighted random selection
                weights = cd[nd_idx]
                weights = np.clip(weights, 1e-10, None)
                weights /= np.sum(weights)
                parent_idx = rng.choice(nd_idx, p=weights)
            else:
                parent_idx = rng.integers(len(X))

            # Mutation with adaptive step size
            # Start with larger steps, decrease over time
            progress = used / max(1, self.budget)
            step_size = 0.3 * (1.0 - 0.7 * progress)
            # Use Cauchy-like distribution for occasional large jumps
            if rng.random() < 0.1:
                step_size *= 3.0

            child = X[parent_idx] + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace worst point in dominated set
            dom_mask = ~nd_mask
            if np.sum(dom_mask) > 0:
                dom_idx = np.where(dom_mask)[0]
                # Pick the worst dominated point (max sum of objectives)
                worst_in_dom = dom_idx[np.argmax(F[dom_idx].sum(axis=1))]
                X[worst_in_dom] = child
                F[worst_in_dom] = fc
            else:
                # All non-dominated; replace the one with lowest crowding distance
                cd = crowding_distances(F, nd_mask)
                worst_idx = np.argmin(cd)
                X[worst_idx] = child
                F[worst_idx] = fc

            # Occasionally, if we've been stuck, do a random restart
            if used % 50 == 0 and used > 50:
                # Replace 20% of population with random points
                n_repl = max(1, pop_size // 5)
                for _ in range(n_repl):
                    if used >= self.budget:
                        break
                    idx = rng.integers(pop_size)
                    X[idx] = lo + span * rng.random(self.dim)
                    F[idx] = problem(X[idx])
                    used += 1
                    if time.time() > deadline:
                        break

        return X
# EVOLVE-BLOCK-END
