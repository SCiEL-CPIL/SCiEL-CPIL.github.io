# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style optimizer with differential evolution mutation for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dominates(self, a, b):
        """Return True if a dominates b (a <= b in all, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                fronts.append([i])
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            for i in fronts[front_idx]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range
        return dist

    def __call__(self, problem, bounds):
        deadline = time.time() + 270.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random
        budget = self.budget

        # Population size: use a moderate size to balance exploration and exploitation
        pop_size = min(30, max(10, budget // 8))
        evals = 0

        # Initialize population using Latin Hypercube Sampling
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            pts = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(pts)
            X[:, d] = lo[d] + pts * span[d]

        for i in range(pop_size):
            F[i] = problem(X[i])
            evals += 1

        # Main evolution loop
        while evals < budget:
            if time.time() > deadline:
                break

            # Generate offspring population
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))

            for i in range(pop_size):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents using tournament selection on non-dominated fronts
                fronts = self._non_dominated_sort(F)

                # Tournament selection: pick 2 random individuals, choose the one in better front
                def tournament_select():
                    a, b = rng.randint(pop_size), rng.randint(pop_size)
                    front_a = min([fi for fi, front in enumerate(fronts) if a in front], default=0)
                    front_b = min([fi for fi, front in enumerate(fronts) if b in front], default=0)
                    if front_a < front_b:
                        return a
                    elif front_b < front_a:
                        return b
                    else:
                        # Same front: use crowding distance
                        cd = self._crowding_distance(F, list(range(pop_size)))
                        return a if cd[a] > cd[b] else b

                p1 = tournament_select()
                p2 = tournament_select()
                p3 = tournament_select()

                # Differential evolution mutation
                # child = p1 + F * (p2 - p3) + C * noise
                F_factor = 0.5 + 0.3 * rng.random()
                C_factor = 0.1 + 0.1 * (1 - evals / budget)

                child = X[p1] + F_factor * (X[p2] - X[p3]) + C_factor * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals += 1

            if evals >= budget:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts = self._non_dominated_sort(combined_F)

            # Select next generation
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - count
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # Descending crowding distance
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            X, F = new_X, new_F

        # Return the best non-dominated set
        fronts = self._non_dominated_sort(F)
        if len(fronts) > 0:
            best_indices = fronts[0]
            return X[best_indices]
        return X
# EVOLVE-BLOCK-END
