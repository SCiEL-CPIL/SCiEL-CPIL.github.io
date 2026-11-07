"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 (2 objectives, 4 variables)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim

        # Budget allocation
        n_init = 20
        remaining = self.budget - n_init

        # Use the seeded random state (evaluator calls np.random.seed(seed))
        # We'll use np.random directly to respect the seed set by evaluator
        # But we also create a local rng for reproducibility within a run
        # Actually, the evaluator seeds np.random, so let's use that for initial diversity
        # and a separate rng for internal operations

        # Initial population: uniform random + a few structured points
        X = np.zeros((n_init, dim))
        # 12 uniform random
        for i in range(12):
            X[i] = lo + span * np.random.random(dim)
        # 4 corner-ish points
        X[12] = lo + 0.1 * span
        X[13] = lo + 0.9 * span
        X[14] = lo + 0.5 * span
        X[15] = (lo + hi) / 2.0
        X[16] = lo + np.array([0.0, 0.5, 0.5, 0.0]) * span
        X[17] = lo + np.array([0.5, 0.0, 0.5, 0.5]) * span
        X[18] = lo + np.array([0.5, 0.5, 0.0, 0.5]) * span
        X[19] = lo + np.array([0.5, 0.5, 0.5, 0.0]) * span

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Track all evaluated points for hypervolume
        all_X = X.copy()
        all_F = F.copy()

        # Non-dominated sort helper
        def non_dominated_front(F_mat):
            """Return indices of non-dominated points in F_mat (N, n_obj)."""
            n = len(F_mat)
            if n == 0:
                return np.array([], dtype=int)
            # A point i is dominated if exists j with F[j] <= F[i] and F[j] != F[i]
            # Vectorized: for each pair
            # F_mat[:, None, :] <= F_mat[None, :, :]  (N, N, n_obj)
            # and any strict
            le = np.all(F_mat[:, None, :] <= F_mat[None, :, :], axis=2)  # N x N
            lt = np.any(F_mat[:, None, :] < F_mat[None, :, :], axis=2)   # N x N
            dominated = np.any(le & lt, axis=0)  # N
            return np.where(~dominated)[0]

        # Crowding distance
        def crowding_distance(F_mat, front_indices):
            """Compute crowding distance for points in front_indices of F_mat."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F_mat[front_indices, m]
                order = np.argsort(obj_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                if obj_vals[order[-1]] - obj_vals[order[0]] > 0:
                    for i in range(1, n - 1):
                        dist[order[i]] += (obj_vals[order[i + 1]] - obj_vals[order[i - 1]]) / (
                            obj_vals[order[-1]] - obj_vals[order[0]])
            return dist

        # Main loop: evolutionary strategy with adaptive mutation
        # Use a small population and iterate
        pop_size = 10
        # Start with best 10 from initial
        front_idx = non_dominated_front(F)
        if len(front_idx) >= pop_size:
            # Take all front + some others
            pop_X = X[front_idx[:pop_size]].copy()
            pop_F = F[front_idx[:pop_size]].copy()
        else:
            # Take front + fill with least dominated
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            n_dom = np.sum(dominated)
            # Sort by number of dominators
            dom_count = np.array([np.sum(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            order = np.argsort(dom_count)
            pop_X = X[order[:pop_size]].copy()
            pop_F = F[order[:pop_size]].copy()

        # Adaptive mutation scale
        mut_scale = 0.15

        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Get non-dominated front of current population
            front = non_dominated_front(pop_F)
            if len(front) == 0:
                front = np.arange(len(pop_F))

            # Crowding distance on front
            if len(front) > 1:
                cd = crowding_distance(pop_F, front)
                # Select parent from front with highest crowding distance
                parent_idx = front[np.argmax(cd)]
            else:
                parent_idx = front[0] if len(front) > 0 else 0

            # Mutation: Gaussian with adaptive scale
            # Decay mutation scale over time
            progress = used / self.budget
            current_scale = mut_scale * (1.0 - 0.5 * progress)

            child = pop_X[parent_idx].copy()
            child += current_scale * span * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Also try a few local perturbations for diversity
            fc = problem(child)
            used += 1
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])

            # Replace worst in population
            # Use a simple dominance-based replacement
            # Find worst: most dominated, or highest sum if tie
            pop_F_new = np.vstack([pop_F, fc])
            pop_X_new = np.vstack([pop_X, child])
            n_new = len(pop_F_new)
            # Compute dominance count
            le = np.all(pop_F_new[:, None, :] <= pop_F_new[None, :, :], axis=2)
            lt = np.any(pop_F_new[:, None, :] < pop_F_new[None, :, :], axis=2)
            dom_count = np.sum(le & lt, axis=1)  # number of points dominated by each
            # Actually, we want to find points that are dominated by many others
            # Let's use: for each point, how many others dominate it
            dom_by = np.sum(np.all(pop_F_new[None, :, :] <= pop_F_new[:, None, :], axis=2) &
                            np.any(pop_F_new[None, :, :] < pop_F_new[:, None, :], axis=2), axis=1)
            worst_idx = np.argmax(dom_by)
            if dom_by[worst_idx] == 0:
                # All non-dominated, replace one with highest crowding distance inverse
                # Just replace a random one
                worst_idx = np.random.randint(n_new)
            pop_X = np.delete(pop_X_new, worst_idx, axis=0)
            pop_F = np.delete(pop_F_new, worst_idx, axis=0)

            # Periodically refresh with a new random point to maintain diversity
            if used % 25 == 0 and used < self.budget - 10:
                new_pt = lo + span * np.random.random(dim)
                fn = problem(new_pt)
                used += 1
                all_X = np.vstack([all_X, new_pt])
                all_F = np.vstack([all_F, fn])
                # Insert into population if it's non-dominated or replaces worst
                pop_F_temp = np.vstack([pop_F, fn])
                pop_X_temp = np.vstack([pop_X, new_pt])
                n_t = len(pop_F_temp)
                dom_by_t = np.sum(np.all(pop_F_temp[None, :, :] <= pop_F_temp[:, None, :], axis=2) &
                                  np.any(pop_F_temp[None, :, :] < pop_F_temp[:, None, :], axis=2), axis=1)
                worst_t = np.argmax(dom_by_t)
                if dom_by_t[worst_t] == 0:
                    worst_t = np.random.randint(n_t)
                pop_X = np.delete(pop_X_temp, worst_t, axis=0)
                pop_F = np.delete(pop_F_temp, worst_t, axis=0)

        return all_X
# EVOLVE-BLOCK-END
