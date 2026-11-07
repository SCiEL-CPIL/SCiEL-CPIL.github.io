"""Multi-objective optimizer for WFG4 using fast NSGA-II with vectorized ops."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast NSGA-II with vectorized dominance for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominance_ranks(self, F):
        """Compute dominance ranks using vectorized operations."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        # Vectorized dominance check
        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)
        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)
        dom = leq & lt
        np.fill_diagonal(dom, False)

        S = [set() for _ in range(n)]
        n_dom = np.sum(dom, axis=0)

        for i in range(n):
            for j in range(n):
                if dom[i, j]:
                    S[i].add(j)

        rank = np.full(n, -1, dtype=int)
        current_front = list(np.where(n_dom == 0)[0])
        if not current_front:
            return np.zeros(n, dtype=int)

        rank_level = 0
        while current_front:
            next_front = []
            for i in current_front:
                rank[i] = rank_level
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            rank_level += 1
            current_front = next_front

        for i in range(n):
            if rank[i] == -1:
                rank[i] = rank_level

        return rank

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points in a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        F_sub = F[indices]
        cd = np.zeros(n)

        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_min = F_sub[order[0], m]
            f_max = F_sub[order[-1], m]
            if f_max - f_min < 1e-12:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (f_max - f_min)

        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 265  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget

        # Smaller population for speed, more generations
        pop_size = 24

        # Initial population
        X = lo + (hi - lo) * rng.random_sample((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        while used < budget:
            if time.time() > deadline:
                break

            # Compute ranks
            ranks = self._dominance_ranks(F)

            # Group by front
            fronts = {}
            for i in range(pop_size):
                r = ranks[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)

            # Crowding distance
            cd_by_rank = {}
            for r, idxs in fronts.items():
                cd_by_rank[r] = self._crowding_distance(F, idxs)

            # Fitness
            fitness = np.zeros(pop_size)
            for r, idxs in fronts.items():
                cd_vals = cd_by_rank[r]
                for k, idx in enumerate(idxs):
                    fitness[idx] = r * 10000.0 - cd_vals[k]

            # Generate offspring
            offspring_X = []
            offspring_F = []

            for _ in range(pop_size):
                if time.time() > deadline:
                    break
                if used + 2 > budget:
                    break

                # Tournament selection
                t1, t2 = rng.randint(pop_size), rng.randint(pop_size)
                p1 = t1 if fitness[t1] < fitness[t2] else t2
                t1, t2 = rng.randint(pop_size), rng.randint(pop_size)
                p2 = t1 if fitness[t1] < fitness[t2] else t2

                # SBX crossover (vectorized)
                child1 = X[p1].copy()
                child2 = X[p2].copy()

                if rng.random() < 0.9:
                    u = rng.random_sample(dim)
                    beta = np.ones(dim)
                    mask = u <= 0.5
                    beta[mask] = (2 * u[mask]) ** (1.0 / 21.0)
                    beta[~mask] = (1.0 / (2 * (1 - u[~mask]))) ** (1.0 / 21.0)
                    child1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                    child2 = 0.5 * ((1 - beta) * X[p1] + (1 + beta) * X[p2])

                # Polynomial mutation (vectorized)
                u = rng.random_sample(dim)
                mask = u < 1.0 / dim
                if np.any(mask):
                    u_m = rng.random_sample(np.sum(mask))
                    delta = np.zeros(dim)
                    delta[mask] = (2 * u_m) ** (1.0 / 21.0) - 1
                    child1 = np.clip(child1 + (hi - lo) * delta, lo, hi)

                u2 = rng.random_sample(dim)
                mask2 = u2 < 1.0 / dim
                if np.any(mask2):
                    u_m2 = rng.random_sample(np.sum(mask2))
                    delta2 = np.zeros(dim)
                    delta2[mask2] = 1 - (2 * u_m2) ** (1.0 / 21.0)
                    child2 = np.clip(child2 + (hi - lo) * delta2, lo, hi)

                offspring_X.append(child1)
                offspring_X.append(child2)

            # Evaluate offspring
            for child in offspring_X:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                fc = problem(child)
                used += 1
                offspring_F.append(fc)

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X[:len(offspring_F)], dtype=np.float64)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_comb = len(combined_F)

            combined_ranks = self._dominance_ranks(combined_F)
            combined_fronts = {}
            for i in range(n_comb):
                r = combined_ranks[i]
                if r not in combined_fronts:
                    combined_fronts[r] = []
                combined_fronts[r].append(i)

            sorted_fronts = sorted(combined_fronts.items(), key=lambda x: x[0])

            new_X = []
            new_F = []
            for r, idxs in sorted_fronts:
                if len(new_X) + len(idxs) <= pop_size:
                    for idx in idxs:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    needed = pop_size - len(new_X)
                    cd_vals = self._crowding_distance(combined_F, idxs)
                    order = np.argsort(-cd_vals)
                    for k in range(min(needed, len(order))):
                        idx = idxs[order[k]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X, dtype=np.float64)
            F = np.array(new_F, dtype=np.float64)

        return X
# EVOLVE-BLOCK-END
