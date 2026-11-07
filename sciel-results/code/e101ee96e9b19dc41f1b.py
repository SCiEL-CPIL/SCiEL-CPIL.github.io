"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 100
        if budget < N:
            N = max(2, budget)

        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Fast non-dominated sorting
        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_dom = [0] * n  # number of solutions dominated
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1] if len(fronts) > 1 else fronts

        # Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            m = len(F[indices[0]])
            cd = {i: 0.0 for i in indices}
            for m_idx in range(m):
                sorted_idx = sorted(indices, key=lambda i: F[i][m_idx])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][m_idx]
                f_max = F[sorted_idx[-1]][m_idx]
                if f_max - f_min < 1e-12:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[k]] += (F[sorted_idx[k + 1]][m_idx] - F[sorted_idx[k - 1]][m_idx]) / (f_max - f_min)
            return cd

        # Binary tournament selection
        def tournament_select(fronts, cd_map, n_select):
            selected = []
            for _ in range(n_select):
                i1 = rng.integers(len(fronts[0]) + sum(len(f) for f in fronts[1:])) if len(fronts) > 1 else rng.integers(len(fronts[0]))
                # Simpler: pick from all solutions
                i1 = rng.integers(N)
                i2 = rng.integers(N)
                # Compare by front rank then crowding
                # Need to assign ranks
                selected.append(i1 if rng.random() < 0.5 else i2)
            return selected

        # Better: use proper NSGA-II selection
        def select_parents(fronts, cd_map, n_select):
            # Assign ranks
            rank = np.zeros(N, dtype=int)
            for r, front in enumerate(fronts):
                for i in front:
                    rank[i] = r
            selected = []
            for _ in range(n_select):
                i1 = rng.integers(N)
                i2 = rng.integers(N)
                if rank[i1] != rank[i2]:
                    selected.append(i1 if rank[i1] < rank[i2] else i2)
                else:
                    cd1 = cd_map.get(i1, 0)
                    cd2 = cd_map.get(i2, 0)
                    selected.append(i1 if cd1 > cd2 else i2)
            return selected

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        beta = 1.0 / (2.0 * rng.random() + 1e-12)
                    else:
                        beta = 1.0 / (2.0 * (1.0 - rng.random()) + 1e-12)
                    beta += 1.0
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child1[d] = 0.5 * ((1 + alpha) * x1[d] + (1 - alpha) * x2[d])
                        else:
                            child1[d] = 0.5 * ((1 - alpha) * x1[d] + (1 + alpha) * x2[d])
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child2[d] = 0.5 * ((1 + alpha) * x2[d] + (1 - alpha) * x1[d])
                        else:
                            child2[d] = 0.5 * ((1 - alpha) * x2[d] + (1 + alpha) * x1[d])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Polynomial mutation
        def polynomial_mutate(x, eta=20.0, prob=1.0 / dim):
            child = x.copy()
            for d in range(dim):
                if rng.random() < prob:
                    delta1 = (child[d] - lo[d]) / span[d]
                    delta2 = (hi[d] - child[d]) / span[d]
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 + (1.0 + 2.0 * delta1) * rand
                        deltaq = (xy ** (1.0 / (eta + 1.0)) - 1.0)
                    else:
                        xy = 1.0 + (1.0 + 2.0 * delta2) * (1.0 - rand)
                        deltaq = 1.0 - (xy ** (1.0 / (eta + 1.0)))
                    child[d] += deltaq * span[d]
                    child[d] = np.clip(child[d], lo[d], hi[d])
            return child

        # Main loop
        while used < budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(N, budget - used)
            for _ in range(n_offspring):
                p1_idx, p2_idx = select_parents(fronts, cd_map, 2)
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                if used >= budget:
                    break
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1
            if len(offspring_X) == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            n_combined = len(combined_F)

            # Non-dominated sort
            fronts = fast_nondominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    for i in front:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                else:
                    # Sort by crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = N - len(next_X)
                    for i in sorted_front[:needed]:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                    break

            X = np.array(next_X)
            F = np.array(next_F)
            N = len(X)

            # Recompute fronts and crowding for next iteration
            fronts = fast_nondominated_sort(F)
            # Compute crowding for all fronts for selection
            cd_map = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, c in cd.items():
                    cd_map[i] = c

        return X
# EVOLVE-BLOCK-END
