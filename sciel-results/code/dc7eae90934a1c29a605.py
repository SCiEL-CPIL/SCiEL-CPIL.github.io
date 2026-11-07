"""Multi-objective optimizer for RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with deadline safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        pop_size = self.pop_size

        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)
            
            # Crowding distance
            all_indices = list(range(len(F)))
            cd = self._crowding_distance(F, all_indices)

            # Selection: tournament
            children = []
            while len(children) < pop_size:
                p1 = self._tournament(F, fronts, cd, rng)
                p2 = self._tournament(F, fronts, cd, rng)
                # Simple crossover (SBX-like or blend)
                alpha = rng.random()
                c1 = alpha * X[p1] + (1 - alpha) * X[p2]
                c2 = (1 - alpha) * X[p1] + alpha * X[p2]
                children.append(c1)
                children.append(c2)
                if len(children) >= pop_size:
                    children = children[:pop_size]

            # Mutation
            for i in range(len(children)):
                if rng.random() < 0.5:
                    children[i] = children[i] + 0.1 * (hi - lo) * rng.standard_normal(d)
                children[i] = np.clip(children[i], lo, hi)

            # Evaluate children
            F_children = np.array([problem(x) for x in children])
            used += len(children)

            # Combine parents and children
            X_combined = np.vstack([X, children])
            F_combined = np.vstack([F, F_children])

            # Select next generation
            next_fronts = self._non_dominated_sort(F_combined)
            next_X = []
            next_F = []
            for front in next_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Sort by crowding distance and take the rest
                    remaining = pop_size - len(next_X)
                    front_cd = [self._crowding_distance(F_combined, [idx])[0] for idx in front]
                    sorted_front = sorted(zip(front, front_cd), key=lambda x: -x[1])
                    for idx, _ in sorted_front[:remaining]:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n

        cd = np.zeros(n)
        m = F.shape[1]

        for k in range(m):
            obj = F[indices, k]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            range_val = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if range_val == 0:
                continue
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / range_val

        return cd

    def _tournament(self, F, fronts, cd, rng, k=2):
        # Map indices to front number
        front_num = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                front_num[idx] = i

        candidates = rng.integers(len(F), size=k)
        best = candidates[0]
        for c in candidates[1:]:
            if front_num[c] < front_num[best]:
                best = c
            elif front_num[c] == front_num[best]:
                # Use crowding distance
                if cd[c] > cd[best]:
                    best = c
        return best
# EVOLVE-BLOCK-END
