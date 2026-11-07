"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust NSGA-II multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_sort(self, F):
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].add(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].add(i)
                    domination_count[i] += 1
        front = np.where(domination_count == 0)[0]
        fronts = [front]
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if len(next_front) == 0:
                break
            fronts.append(np.array(next_front))
            front = np.array(next_front)
        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(42)  # Fixed seed for reproducibility within the run, but evaluator seeds np.random
        # Note: The evaluator calls np.random.seed(seed). We should use np.random directly or derive from it?
        # The task says "evaluator calls np.random.seed(seed)". Using np.random.default_rng(0) ignores that.
        # It is better to use np.random directly to respect the evaluator's seeding.
        rng = np.random

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        pop_size = 30
        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        eval_count = pop_size

        while eval_count < self.budget:
            if time.time() > deadline:
                break

            # Calculate non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                rank[front] = i + 1

            # Calculate crowding distance
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F, front)
                else:
                    cd[front] = np.inf

            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Tournament Selection
                idx1 = rng.randint(pop_size)
                idx2 = rng.randint(pop_size)
                if rank[idx1] < rank[idx2]:
                    p1 = idx1
                elif rank[idx2] < rank[idx1]:
                    p1 = idx2
                else:
                    p1 = idx1 if cd[idx1] > cd[idx2] else idx2

                idx3 = rng.randint(pop_size)
                idx4 = rng.randint(pop_size)
                if rank[idx3] < rank[idx4]:
                    p2 = idx3
                elif rank[idx4] < rank[idx3]:
                    p2 = idx4
                else:
                    p2 = idx3 if cd[idx3] > cd[idx4] else idx4

                # SBX Crossover
                child = X[p1].copy()
                if rng.random() < 0.9:
                    for j in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 11)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 11)
                            c1 = 0.5 * ((1 + beta) * X[p1, j] + (1 - beta) * X[p2, j])
                            child[j] = c1

                # Polynomial Mutation
                if rng.random() < 0.9:
                    for j in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            delta1 = (rng.random() * 2 - 1)
                            delta = (1 + 2 * abs(delta1)) ** (-1 / 21)
                            if delta1 >= 0:
                                delta = (1 - delta)
                            else:
                                delta = (delta - 1)
                            child[j] += delta * range_[j]

                child = np.clip(child, lo, hi)
                fc = problem(child)
                eval_count += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if len(offspring_X) == 0:
                break

            # Combine
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])

            # Select next population
            fronts = self._non_dominated_sort(F_combined)
            selected = np.array([], dtype=int)
            for i, front in enumerate(fronts):
                if len(selected) + len(front) <= pop_size:
                    selected = np.concatenate([selected, front])
                else:
                    remaining = pop_size - len(selected)
                    cd_front = self._crowding_distance(F_combined, front)
                    sorted_front = front[np.argsort(cd_front)[::-1]]
                    selected = np.concatenate([selected, sorted_front[:remaining]])
                    break

            X = X_combined[selected]
            F = F_combined[selected]

        return X
# EVOLVE-BLOCK-END
