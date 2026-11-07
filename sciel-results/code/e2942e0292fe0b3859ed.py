"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim

        # Determine population size
        pop_size = 24
        # Total evaluations: pop_size (init) + pop_size * n_gen (offspring)
        # We want pop_size + pop_size * n_gen <= budget
        n_gen = max(1, (self.budget - pop_size) // pop_size)

        # Initial population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[j] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1

            k = 0
            while front[k] > 0:
                next_front = []
                for i in range(n):
                    if rank[i] == k:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                rank[j] = k + 1
                                next_front.append(j)
                k += 1
                if next_front:
                    front.append(len(next_front))
                else:
                    break
            return rank

        # Helper: crowding distance
        def crowding_distance(F, rank):
            n = len(F)
            cd = [0.0] * n
            for r in set(rank):
                members = [i for i in range(n) if rank[i] == r]
                if len(members) <= 2:
                    for i in members:
                        cd[i] = float('inf')
                    continue
                for m in range(n_obj):
                    vals = [(F[i][m], i) for i in members]
                    vals.sort()
                    cd[vals[0][1]] = float('inf')
                    cd[vals[-1][1]] = float('inf')
                    min_val = vals[0][0]
                    max_val = vals[-1][0]
                    if max_val - min_val < 1e-12:
                        continue
                    for k in range(1, len(vals) - 1):
                        cd[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / (max_val - min_val)
            return cd

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if abs(x1[i] - x2[i]) < 1e-14:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            x = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    lo_i = lo[i]
                    hi_i = hi[i]
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta_q * (hi_i - lo_i)
                    x[i] = np.clip(x[i], lo_i, hi_i)
            return x

        # Binary tournament selection
        def tournament_select(X, F, rank, cd, k=2):
            idxs = rng.integers(0, len(F), size=k)
            best = idxs[0]
            for idx in idxs[1:]:
                if rank[idx] < rank[best] or (rank[idx] == rank[best] and cd[idx] > cd[best]):
                    best = idx
            return best

        for gen in range(n_gen):
            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < pop_size and used < self.budget:
                p1 = tournament_select(X, F, rank, cd)
                p2 = tournament_select(X, F, rank, cd)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                if used < self.budget:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                else:
                    break

            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort
            rank_all = non_dominated_sort(F_all)
            cd_all = crowding_distance(F_all, rank_all)

            # Select next generation
            next_X = []
            next_F = []
            i = 0
            while i < len(rank_all) and len(next_X) < pop_size:
                if rank_all[i] < 0:
                    i += 1
                    continue
                members = [j for j in range(len(rank_all)) if rank_all[j] == rank_all[i]]
                if len(next_X) + len(members) <= pop_size:
                    for j in members:
                        next_X.append(X_all[j])
                        next_F.append(F_all[j])
                else:
                    # Sort by crowding distance
                    members.sort(key=lambda j: cd_all[j], reverse=True)
                    for j in members[:pop_size - len(next_X)]:
                        next_X.append(X_all[j])
                        next_F.append(F_all[j])
                    break
                i += 1

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
