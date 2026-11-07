"""Multi-objective optimizer on ZDT3."""
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(100, max(20, budget // 2))

        # Initial population: stratified sampling for diversity
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # Use a mix of random and structured initialization
            if i == 0:
                X[i] = lo + (hi - lo) * 0.5
            elif i == 1:
                X[i] = lo + (hi - lo) * 0.1
            elif i == 2:
                X[i] = lo + (hi - lo) * 0.9
            else:
                X[i] = lo + (hi - lo) * rng.random(dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Crowding distance
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.ones(n) * np.inf
            cd = np.zeros(n)
            for m in range(n_obj):
            # Sort by objective m
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F[order[-1], m] - F[order[0], m] == 0:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (
                        F[order[-1], m] - F[order[0], m]
                    )
            return cd

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    front.append(i)

            fronts = [front]
            while fronts[-1]:
                next_front = []
                for i in fronts[-1]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
            return fronts

        # Binary tournament selection
        def tournament_select(X, F, fronts, cd):
            i = rng.integers(len(X))
            j = rng.integers(len(X))
            # Compare by front rank, then crowding distance
            front_i = next(k for k, f in enumerate(fronts) if i in f)
            front_j = next(k for k, f in enumerate(fronts) if j in f)
            if front_i < front_j:
                return X[i]
            elif front_j < front_i:
                return X[j]
            else:
                if cd[i] >= cd[j]:
                    return X[i]
                else:
                    return X[j]

        # Crossover (SBX)
        def sbx_crossover(x1, x2, lo, hi):
            if rng.random() < 0.9:
                eta = 20.0
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    if rng.random() < 0.5:
                        if abs(x1[i] - x2[i]) < 1e-14:
                            child1[i] = x1[i]
                            child2[i] = x2[i]
                        else:
                            u = rng.random()
                            if u <= 1.0 / (eta + 1):
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            elif u >= 1.0 - 1.0 / (eta + 1):
                                beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                            else:
                                beta = 1.0
                            child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                            child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                            child1[i] = np.clip(child1[i], lo[i], hi[i])
                            child2[i] = np.clip(child2[i], lo[i], hi[i])
                    else:
                        child1[i] = x2[i]
                        child2[i] = x1[i]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        # Mutation (polynomial)
        def polynomial_mutation(x, lo, hi):
            eta = 20.0
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x[i] = x[i] + delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            cd = crowding_distance(F)

            # Generate offspring
            offspring = []
            while len(offspring) < pop_size and used + len(offspring) < budget:
                p1 = tournament_select(X, F, fronts, cd)
                p2 = tournament_select(X, F, fronts, cd)
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                offspring.append(c1)
                offspring.append(c2)
                used += 2

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])

            # Combine parents and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])

            # Sort combined population
            fronts_all = non_dominated_sort(F_all)
            cd_all = crowding_distance(F_all)

            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_all:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Sort by crowding distance
                    front_cd = [cd_all[idx] for idx in front]
                    order = np.argsort(front_cd)[::-1]
                    need = pop_size - len(X_new)
                    for k in range(min(need, len(order))):
                        idx = front[order[k]]
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return X
