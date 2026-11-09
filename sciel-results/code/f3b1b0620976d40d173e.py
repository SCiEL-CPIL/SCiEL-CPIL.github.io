"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        pop_size = 30
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    fronts.append([i])
            front_idx = 0
            while front_idx < len(fronts):
                next_front = []
                for i in fronts[front_idx]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front_idx += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            cd = {i: 0.0 for i in indices}
            F_sub = F[indices]
            for m in range(self.n_obj):
                sorted_idx = sorted(indices, key=lambda i: F[i, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min > 1e-15:
                    for k in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / (f_max - f_min)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            for _ in range(pop_size):
                # Tournament selection from non-dominated front
                fronts = non_dominated_sort(F)
                if not fronts:
                    break
                # Pick from first front
                front = fronts[0]
                if len(front) > 1:
                    cd = crowding_distance(F, front)
                    # Select 2 parents via tournament
                    p1 = front[rng.integers(len(front))]
                    p2 = front[rng.integers(len(front))]
                else:
                    p1 = p2 = front[0]

                # Simulated binary crossover
                child1 = X[p1].copy()
                child2 = X[p2].copy()
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        beta = 1.0 / (1.0 + 2.0 * min(X[p1, d], X[p2, d]) / max(abs(X[p2, d] - X[p1, d]), 1e-15))
                        eta = rng.random()
                        if eta <= 0.5:
                            beta_prime = beta ** (1.0 / (2.0 + 1.0))
                        else:
                            beta_prime = (1.0 / (2.0 - beta)) ** (1.0 / (2.0 + 1.0))
                        child1[d] = 0.5 * ((1.0 + beta_prime) * X[p1, d] + (1.0 - beta_prime) * X[p2, d])
                        child2[d] = 0.5 * ((1.0 + 1.0 / beta_prime) * X[p1, d] + (1.0 - 1.0 / beta_prime) * X[p2, d])

                # Adaptive mutation
                mutation_rate = 0.5
                sigma = 0.1 * (hi - lo)
                for d in range(self.dim):
                    if rng.random() < mutation_rate:
                        child1[d] += sigma * rng.standard_normal()
                    if rng.random() < mutation_rate:
                        child2[d] += sigma * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                if used < self.budget:
                    offspring_X.append(child1)
                    offspring_F.append(problem(child1))
                    used += 1
                if used < self.budget:
                    offspring_X.append(child2)
                    offspring_F.append(problem(child2))
                    used += 1
                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
