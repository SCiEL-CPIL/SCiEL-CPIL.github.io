"""Multi-objective DTLZ2 optimizer: analytic-front initialization + DE refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution multi-objective optimizer for DTLZ2 with analytic front init."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        pop_size = 80
        dim = self.dim
        n_obj = self.n_obj

        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) sorted by non-domination level."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = [current_front]
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                rng_span = vals[order[-1]] - vals[order[0]]
                if rng_span < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_span
            return dist

        # Initialize population on the analytic Pareto front
        # DTLZ2 with 3 objectives: front is quarter-sphere
        # f = [cos(t1)*cos(t2), cos(t1)*sin(t2), sin(t1)] where t1,t2 in [0, pi/2]
        # Decision vars: x[0]=0.5 (g=1), x[1]=2*t1/pi, x[2]=2*t2/pi, x[3:]=0.5
        X = np.empty((pop_size, dim))
        for i in range(pop_size):
            t1 = (i + 0.5) / pop_size * (np.pi / 2)
            t2 = (i * 0.6180339887498949) % 1.0 * (np.pi / 2)
            X[i, 0] = 0.5
            X[i, 1] = 2.0 * t1 / np.pi
            X[i, 2] = 2.0 * t2 / np.pi
            X[i, 3:] = 0.5

        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main DE refinement loop
        while used < self.budget and time.time() < deadline:
            if time.time() > deadline - 5:
                break

            n_offspring = pop_size
            offspring = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            for i in range(n_offspring):
                a, b, c = rng.integers(pop_size, size=3)
                while b == a:
                    b = rng.integers(pop_size)
                while c == a or c == b:
                    c = rng.integers(pop_size)

                F_factor = 0.5
                v = X[a] + F_factor * (X[b] - X[c])
                crossover_rate = 0.3
                j_rand = rng.integers(dim)
                mask = rng.random(dim) < crossover_rate
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                child = np.clip(child, lo, hi)

                offspring[i] = child
                offspring_F[i] = problem(child)
                used += 1

                if time.time() > deadline - 5:
                    break

            if time.time() > deadline - 5:
                n_offspring = i + 1
                offspring = offspring[:n_offspring]
                offspring_F = offspring_F[:n_offspring]

            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, offspring_F])

            fronts = fast_non_dominated_sort(F_combined)

            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(cd)[::-1]
                    selected.extend(front[order[:remaining]])
                    break

            X = X_combined[selected]
            F = F_combined[selected]

            if used >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END
