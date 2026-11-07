# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size: use a good fraction of budget
        pop_size = min(60, self.budget // 3)
        pop_size = max(pop_size, 20)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            dist = np.zeros(len(indices))
            for m in range(self.n_obj):
                vals = F[indices, m]
                idx_sorted = np.argsort(vals)
                dist[idx_sorted[0]] = np.inf
                dist[idx_sorted[-1]] = np.inf
                min_val = vals[idx_sorted[0]]
                max_val = vals[idx_sorted[-1]]
                if max_val - min_val < 1e-30:
                    continue
                for k in range(1, len(indices) - 1):
                    dist[idx_sorted[k]] += (vals[idx_sorted[k + 1]] - vals[idx_sorted[k - 1]]) / (max_val - min_val)
            return dist

        def select_parents(F, X, n_sel):
            """Select parents using tournament selection based on non-domination rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Compute crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_vals[i]
            # Tournament selection
            selected = []
            for _ in range(n_sel):
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if rank[a] < rank[b]:
                    winner = a
                elif rank[b] < rank[a]:
                    winner = b
                else:
                    winner = a if cd[a] > cd[b] else b
                selected.append(winner)
            return selected

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            x = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    beta = 1.0 / (1.0 + 2.0 * rng.random())
                    beta_q = (0.5 * (1.0 + beta)) ** (1.0 / (eta + 1.0))
                    x[d] = 0.5 * ((1.0 + beta_q) * x1[d] + (1.0 - beta_q) * x2[d])
                else:
                    beta = 1.0 / (1.0 + 2.0 * rng.random())
                    beta_q = (0.5 * (1.0 + 1.0 / beta)) ** (1.0 / (eta + 1.0))
                    x[d] = 0.5 * ((1.0 + beta_q) * x2[d] + (1.0 - beta_q) * x1[d])
            return x

        def mutate(x):
            """Polynomial mutation."""
            eta = 20.0
            xm = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.9:  # mutation probability per gene
                    lo_d, hi_d = lo[d], hi[d]
                    if hi_d - lo_d < 1e-10:
                        continue
                    y = x[d]
                    delta1 = (y - lo_d) / (hi_d - lo_d)
                    delta2 = (hi_d - y) / (hi_d - lo_d)
                    rand = rng.random()
                    if rand < 0.5:
                        xy = (2.0 * rand + (1.0 - 2.0 * rand) * (delta1 ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                        val = y - (hi_d - lo_d) * (1.0 - xy)
                    else:
                        xy = (2.0 * (1.0 - rand) + (2.0 * rand - 1.0) * (delta2 ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                        val = y + (hi_d - lo_d) * (1.0 - xy)
                    xm[d] = np.clip(val, lo_d, hi_d)
            return xm

        # Main evolution loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Select parents
            n_children = pop_size
            parents = select_parents(F, X, 2 * n_children)
            
            # Generate offspring
            offspring_X = []
            for i in range(n_children):
                p1 = parents[2 * i]
                p2 = parents[2 * i + 1]
                child = crossover(X[p1], X[p2])
                child = mutate(child)
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_children

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Truncate to population size using non-dominated sorting and crowding distance
            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd_vals = crowding_distance(combined_F, front)
                    sorted_idx = np.argsort(-cd_vals)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_idx[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
