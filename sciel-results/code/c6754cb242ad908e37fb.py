# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        N = 40  # population size
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_doms = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1

            front = [i for i in range(n) if n_doms[i] == 0]
            if not front:
                return [list(range(n))]

            fronts.append(front)
            current_front = front
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front

            # Any remaining points (shouldn't happen)
            assigned = set()
            for f in fronts:
                assigned.update(f)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for points at given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_v = vals.min()
                max_v = vals.max()
                if max_v == min_v:
                    dist[:] = np.inf
                    continue
                # Sort by this objective
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
            return dist

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Evaluate offspring
            # Use SBX-like crossover and mutation
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)

            for i in range(N):
                # Tournament selection for two parents
                def tournament(pop_F, pop_X, k=2):
                    candidates = rng.integers(0, len(pop_F), size=k)
                    best = candidates[0]
                    for c in candidates[1:]:
                        # Compare via dominance
                        a, b = pop_F[c], pop_F[best]
                        if np.all(a <= b) and np.any(a < b):
                            best = c
                        elif not (np.all(b <= a) and np.any(b < a)):
                            # Incomparable, pick random
                            if rng.random() < 0.5:
                                best = c
                    return pop_X[best]

                p1 = tournament(F, X)
                p2 = tournament(F, X)

                # Simulated binary crossover
                if rng.random() < 0.9:
                    child1 = np.empty(self.dim)
                    child2 = np.empty(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            beta = 1.0
                            if rng.random() <= 0.5:
                                beta = (2 * rng.random()) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - rng.random()))) ** (1 / (2 + 1))
                            c1 = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                            c2 = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                            child1[d] = c1
                            child2[d] = c2
                        else:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Mutation
                mutation_rate = 1.0 / self.dim
                for d in range(self.dim):
                    if rng.random() < mutation_rate:
                        sigma = 0.1 * (hi[d] - lo[d])
                        child1[d] = np.clip(child1[d] + rng.normal(0, sigma), lo[d], hi[d])
                    if rng.random() < mutation_rate:
                        sigma = 0.1 * (hi[d] - lo[d])
                        child2[d] = np.clip(child2[d] + rng.normal(0, sigma), lo[d], hi[d])

                offspring_X[i * 2] = child1
                offspring_X[i * 2 + 1] = child2

                f1 = problem(child1)
                f2 = problem(child2)
                offspring_F[i * 2] = f1
                offspring_F[i * 2 + 1] = f2
                used += 2

                if used >= self.budget:
                    break

            if used >= self.budget:
                break

            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            all_indices = np.arange(2 * N)

            # Non-dominated sort
            fronts = fast_non_dominated_sort(all_F)

            # Select next population
            new_X = np.empty_like(X)
            new_F = np.empty_like(F)
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    for idx in front:
                        new_X[count] = all_X[idx]
                        new_F[count] = all_F[idx]
                        count += 1
                else:
                    # Need to select some from this front
                    remaining = N - count
                    cd = crowding_distance(all_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    for k in range(remaining):
                        idx = front[order[k]]
                        new_X[count] = all_X[idx]
                        new_F[count] = all_F[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
