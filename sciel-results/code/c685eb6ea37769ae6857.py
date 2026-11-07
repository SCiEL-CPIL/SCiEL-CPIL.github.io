"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3.0, 5.0, 7.0])

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1

        current_front = [i for i in range(n) if domination_count[i] == 0]
        if not current_front:
            return [list(range(n))]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        assigned = set()
        for f in fronts:
            for i in f:
                assigned.add(i)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)

        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        obj = F[indices]

        for m in range(self.n_obj):
            sorted_idx = np.argsort(obj[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (obj[sorted_idx[k + 1], m] - obj[sorted_idx[k - 1], m]) / obj_range

        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        pop_size = 15
        if self.budget < 100:
            pop_size = max(10, self.budget // 5)

        # WFG4 has a known structure: x1 = 0.35 is critical for the global Pareto front.
        # Initialize with a mix of random points and points near x1=0.35.
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        
        # Place 30% of initial population near x1 = 0.35
        n_near = max(1, pop_size // 3)
        for i in range(n_near):
            X[i, 0] = 0.35 + (hi[0] - lo[0]) * (rng.random() - 0.5) * 0.1
            X[i, 0] = np.clip(X[i, 0], lo[0], hi[0])

        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        while used + pop_size <= self.budget:
            if time.time() > deadline:
                break

            fronts = self._non_dominated_sort(F)

            # Selection
            mating_pool = []
            front_rank = {}
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_rank[idx] = fi
                if len(mating_pool) + len(front) >= pop_size:
                    cd = self._crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(mating_pool)
                    mating_pool.extend(sorted_front[:needed])
                    break
                else:
                    mating_pool.extend(front)

            def tournament():
                i = rng.integers(len(mating_pool))
                j = rng.integers(len(mating_pool))
                pi = mating_pool[i]
                pj = mating_pool[j]
                ri = front_rank.get(pi, len(fronts))
                rj = front_rank.get(pj, len(fronts))
                if ri < rj:
                    return pi
                elif rj < ri:
                    return pj
                else:
                    return pi if rng.random() < 0.5 else pj

            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring):
                p1 = X[tournament()]
                p2 = X[tournament()]
                
                # SBX crossover
                if rng.random() < 0.9:
                    beta1 = 1.0
                    if rng.random() < 0.5:
                        beta1 = (2.0 * rng.random()) ** (1.0 / 11.0)
                    else:
                        beta1 = (1.0 / (2.0 * rng.random())) ** (1.0 / 11.0)
                    beta2 = 1.0 / beta1
                    child = 0.5 * ((1.0 + beta1) * p1 + (1.0 - beta1) * p2)
                    child2 = 0.5 * ((1.0 + beta2) * p1 + (1.0 - beta2) * p2)
                    if rng.random() < 0.5:
                        child = child
                    else:
                        child = child2
                else:
                    child = p1.copy()
                
                # Polynomial mutation
                if rng.random() < 0.9:
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim:
                            if rng.random() < 0.5:
                                delta1 = (2.0 * rng.random()) ** (1.0 / 21.0)
                                child[d] -= delta1 * (hi[d] - lo[d])
                            else:
                                delta2 = (2.0 * (1.0 - rng.random())) ** (1.0 / 21.0)
                                child[d] += delta2 * (hi[d] - lo[d])
                
                # WFG4-specific: bias x1 toward 0.35
                # With some probability, pull x1 closer to 0.35
                if rng.random() < 0.4:
                    target_x1 = 0.35
                    # Blend current x1 with target
                    blend = rng.random() * 0.5 + 0.2
                    child[0] = (1.0 - blend) * child[0] + blend * target_x1
                    child[0] = np.clip(child[0], lo[0], hi[0])

                # Jump mutation for other dimensions
                if rng.random() < 0.2:
                    n_jumps = rng.integers(1, 3)
                    jump_dims = rng.choice(self.dim, size=n_jumps, replace=False)
                    for d in jump_dims:
                        child[d] = lo[d] + (hi[d] - lo[d]) * rng.random()

                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += n_offspring

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            new_fronts = self._non_dominated_sort(combined_F)

            next_X = []
            next_F = []
            for front in new_fronts:
                if len(next_X) + len(front) >= pop_size:
                    cd = self._crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
                else:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])

            X = np.array(next_X)
            F = np.array(next_F)

        # Use remaining budget for random restarts focused on x1=0.35
        while used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random((1, self.dim))
            x[0, 0] = 0.35 + (hi[0] - lo[0]) * (rng.random() - 0.5) * 0.1
            x[0, 0] = np.clip(x[0, 0], lo[0], hi[0])
            f = problem(x[0])
            X = np.vstack([X, x])
            F = np.vstack([F, f.reshape(1, -1)])
            used += 1

        fronts = self._non_dominated_sort(F)
        best_front = fronts[0] if fronts else [0]
        return X[best_front]
# EVOLVE-BLOCK-END
