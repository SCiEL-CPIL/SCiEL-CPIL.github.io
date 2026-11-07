"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4: NSGA-II style with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(20, self.budget // 4))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts (each front is a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            for k in range(F_sub.shape[1]):
            # Normalize
                min_val = F_sub[:, k].min()
                max_val = F_sub[:, k].max()
                if max_val > min_val:
                    F_norm = (F_sub[:, k] - min_val) / (max_val - min_val)
                else:
                    F_norm = np.zeros(m)
                sorted_idx = np.argsort(F_norm)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += F_norm[sorted_idx[i + 1]] - F_norm[sorted_idx[i - 1]]
            return cd

        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(F, front)
                elif len(front) == 2:
                    cd[front] = np.inf
                else:
                    cd[front] = np.inf

            # Select parents using tournament selection based on rank and crowding distance
            def compare(i, j):
                if ranks[i] < ranks[j]:
                    return True
                if ranks[i] > ranks[j]:
                    return False
                return cd[i] > cd[j]

            def tournament(k=2):
                contenders = rng.integers(len(F), size=k)
                best = contenders[0]
                for c in contenders[1:]:
                    if compare(c, best):
                        best = c
                return best

            # Generate children
            n_children = min(pop_size, self.budget - used)
            children_X = []
            children_F = []

            for _ in range(n_children):
                p1 = tournament()
                p2 = tournament()

                # Simulated binary crossover (SBX)
                child1 = X[p1].copy()
                child2 = X[p2].copy()
                eta_c = 20.0
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                        c1 = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        c2 = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                        child1[d] = c1
                        child2[d] = c2

                # Polynomial mutation
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        child1[d] += delta * (hi[d] - lo[d])
                        if rng.random() < 0.5:
                            u2 = rng.random()
                            if u2 < 0.5:
                                delta2 = (2 * u2) ** (1 / (eta_m + 1)) - 1
                            else:
                                delta2 = 1 - (2 * (1 - u2)) ** (1 / (eta_m + 1))
                            child2[d] += delta2 * (hi[d] - lo[d])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                children_X.append(child1)
                children_F.append(problem(child1))
                used += 1

                if used < self.budget and time.time() < deadline - 5.0:
                    children_X.append(child2)
                    children_F.append(problem(child2))
                    used += 1

            # Combine parents and children
            X_all = np.vstack([X, np.array(children_X)])
            F_all = np.vstack([F, np.array(children_F)])

            # Select next generation
            fronts_all = fast_non_dominated_sort(F_all)
            next_X = []
            next_F = []
            for front in fronts_all:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd_front = crowding_distance(F_all, front)
                    sorted_front = sorted(front, key=lambda i: cd_front[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
