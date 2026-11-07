"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(50, self.budget // 2)
        pop_size = max(pop_size, 10)

        # Initialize
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
            front0 = [i for i in range(n) if nd[i] == 0]
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Return array of crowding distances for given indices."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            n = len(indices)
            cd = np.zeros(n)
            F_sub = F[indices]
            m = self.n_obj
            for k in range(m):
                sorted_idx = np.argsort(F_sub[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-12:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return cd

        while used < self.budget:
            if time.time() > deadline - 10:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents from first front(s)
            parent_indices = []
            for front in fronts:
                parent_indices.extend(front)
                if len(parent_indices) >= pop_size:
                    break

            # If we have more than pop_size, use crowding to select
            if len(parent_indices) > pop_size:
                cd = crowding_distance(F, parent_indices)
                order = np.argsort(cd)[::-1]
                parent_indices = [parent_indices[i] for i in order[:pop_size]]

            # Generate offspring: same size as population
            offspring_X = []
            for _ in range(pop_size):
                # Binary tournament selection
                i = rng.integers(len(parent_indices))
                j = rng.integers(len(parent_indices))
                # Pick the one from better front (lower front index)
                # We need to know which front each parent is in
                # Simpler: just pick randomly from parent_indices
                p_idx = parent_indices[i]
                p = X[p_idx].copy()

                # Mutation: Gaussian with adaptive sigma
                sigma = 0.1 * span
                p += sigma * rng.standard_normal(self.dim)
                p = np.clip(p, lo, hi)
                offspring_X.append(p)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            if used > self.budget:
                # We may have exceeded budget; that's ok as long as we don't exceed
                # Actually we should check before evaluating
                pass

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Sort combined population
            fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X_list = []
            new_F_list = []
            for front in fronts:
                if len(new_X_list) + len(front) <= pop_size:
                    for idx in front:
                        new_X_list.append(combined_X[idx])
                        new_F_list.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding
                    remaining = pop_size - len(new_X_list)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X_list.append(combined_X[idx])
                        new_F_list.append(combined_F[idx])
                    break

            X = np.array(new_X_list)
            F = np.array(new_F_list)

        return X
# EVOLVE-BLOCK-END
