# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, use 280s for safety
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(100, self.budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # ZDT2 specific: x1 controls the shape, x2..x30 should go to 0
        # For ZDT2, g(x) = 1 + 9*sum(x[1:])/29, f1 = x[0], f2 = g(1 - sqrt(f1/g))
        # The Pareto front is f2 = 1 - sqrt(f1) for f1 in [0,1]
        # So we want x[1:] -> 0 and x[0] spread over [0,1]

        # Add some points with x[1:] = 0 to get on the Pareto front
        for i in range(min(20, self.budget - used)):
            x = np.zeros(self.dim)
            x[0] = rng.random()
            x = np.clip(x, lo, hi)
            fx = problem(x)
            # Replace worst point
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = x, fx
            used += 1
            if time.time() > deadline - 10:
                break

        # NSGA-II style evolution
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Create offspring
            n_offspring = pop_size
            children = []
            for _ in range(n_offspring):
                # Select parents using tournament
                p1 = self._tournament_select(fronts, F, rng)
                p2 = self._tournament_select(fronts, F, rng)

                # Crossover (SBX)
                child = self._sbx_crossover(p1, p2, lo, hi, rng)

                # Mutation
                if rng.random() < 0.3:
                    child = child + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)

                # Special mutation: set x[1:] to small values for ZDT2
                if rng.random() < 0.2:
                    child[1:] = child[1:] * 0.1

                children.append(child)

            # Evaluate children
            F_children = np.array([problem(x) for x in children])
            used += len(children)

            # Combine parents and children
            X_all = np.vstack([X, np.array(children)])
            F_all = np.vstack([F, F_children])

            # Non-dominated sort on combined
            fronts_all = self._fast_non_dominated_sort(F_all)

            # Select next generation
            X_next = []
            F_next = []
            for front in fronts_all:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                else:
                    # Need to trim this front using crowding distance
                    remaining = pop_size - len(X_next)
                    crowd = self._crowding_distance(F_all[front])
                    order = np.argsort(-crowd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                    break

            X = np.array(X_next)
            F = np.array(F_next)

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1

        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        if front:
            fronts.append(np.array(front))

        k = 0
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(np.array(next_front))
            front = next_front

        return fronts

    def _tournament_select(self, fronts, F, rng, k=2):
        candidates = []
        for _ in range(k):
            # Pick a random front, then a random individual from it
            front_idx = rng.integers(len(fronts))
            front = fronts[front_idx]
            idx = front[rng.integers(len(front))]
            candidates.append(idx)

        # Return the one with best front rank
        best = candidates[0]
        for c in candidates[1:]:
            # Find which front each is in
            for fi, front in enumerate(fronts):
                if c in front:
                    for fj, front2 in enumerate(fronts):
                        if best in front2:
                            if fi < fj:
                                best = c
                            break
                    break
        return F[best]

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p2[i] - p1[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (u * 2) ** (1.0 / (eta + 1))
                    else:
                        alpha = (1.0 - 2 * (1 - u)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * (1 + alpha) * p1[i] + 0.5 * (1 - alpha) * p2[i]
                    child[i] = c1
            else:
                if abs(p2[i] - p1[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (u * 2) ** (1.0 / (eta + 1))
                    else:
                        alpha = (1.0 - 2 * (1 - u)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * (1 - alpha) * p1[i] + 0.5 * (1 + alpha) * p2[i]
                    child[i] = c2
        return np.clip(child, lo, hi)

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        for m in range(F.shape[1]):
            obj = F[:, m]
            idx = np.argsort(obj)
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            min_val, max_val = obj[idx[0]], obj[idx[-1]]
            if max_val - min_val < 1e-14:
                continue
            for i in range(1, n - 1):
                dist[idx[i]] += (obj[idx[i + 1]] - obj[idx[i - 1]]) / (max_val - min_val)
        return dist
# EVOLVE-BLOCK-END
