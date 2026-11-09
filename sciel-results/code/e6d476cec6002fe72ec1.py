import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        # Crossover probability
        cx_prob = 0.9
        # Mutation: polynomial mutation parameters
        eta_m = 20.0

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (for niche diversity if needed)
        # We'll use NSGA-II style non-dominated sorting + crowding distance

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            k = 0
            while len(fronts[k]) > 0:
                Q = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            Q.append(j)
                if Q:
                    fronts.append(Q)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * 1e10
            dist = np.zeros(len(indices))
            for m in range(self.n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx == mn:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = 1e10
                dist[order[-1]] = 1e10
                for i in range(1, len(order) - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (mx - mn)
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break

                # Select parents using tournament selection
                def tournament_select(X, F, k=2):
                    idxs = rng.integers(0, len(X), size=k)
                    # Pick non-dominated among tournament
                    best = idxs[0]
                    for idx in idxs[1:]:
                        if np.all(F[idx] <= F[best]) and np.any(F[idx] < F[best]):
                            best = idx
                        elif not (np.all(F[best] <= F[idx]) and np.any(F[best] < F[idx])):
                            # Both non-dominated, pick random
                            if rng.random() < 0.5:
                                best = idx
                    return best

                p1_idx = tournament_select(X, F)
                p2_idx = tournament_select(X, F)
                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # Simulated binary crossover
                if rng.random() < cx_prob:
                    # SBX
                    eta_c = 20.0
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for j in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(p1[j] - p2[j]) > 1e-14:
                                if rng.random() < 0.5:
                                    beta = 1 + 2 * (p1[j] - p2[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 1
                                    beta = max(beta, 1e-10)
                                    alpha = 2 - beta ** (-eta_c - 1)
                                    rand = rng.random()
                                    if rand <= 1 / alpha:
                                        beta_q = (rand * alpha) ** (1 / (eta_c + 1))
                                    else:
                                        beta_q = (1 / (2 - rand * alpha)) ** (1 / (eta_c + 1))
                                    child1[j] = 0.5 * ((1 + beta_q) * p1[j] + (1 - beta_q) * p2[j])
                                    child2[j] = 0.5 * ((1 - beta_q) * p1[j] + (1 + beta_q) * p2[j])
                                else:
                                    child1[j] = p2[j]
                                    child2[j] = p1[j]
                            else:
                                child1[j] = p1[j]
                                child2[j] = p2[j]
                        else:
                            child1[j] = p1[j]
                            child2[j] = p2[j]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    offspring_X.append(p1.copy())
                    offspring_X.append(p2.copy())

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)

            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F_comb)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_comb, front)
                    order = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
