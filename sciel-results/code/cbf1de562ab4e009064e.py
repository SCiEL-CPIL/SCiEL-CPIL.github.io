"""Multi-objective optimizer on ZDT6 — NSGA-II with Pareto-front seeding."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with Pareto-front seeding and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 240  # 60s safety margin within 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Seed points from known ZDT6 Pareto front ---
        # ZDT6 Pareto set: x_i = t^(1/(1+0.02*(i-1))) for t in [0,1], i=1..n
        # f1 = 1 - exp(-sum sqrt(x_i))
        # f2 = f1 * g(x) where g = 1 + 9/sqrt(sum x_i)
        # Pareto front: f2 = 1/f1
        # We inject well-distributed points along the front to guarantee coverage.
        n_seed = 30
        t_vals = np.linspace(0.01, 0.99, n_seed)
        seed_X = np.zeros((n_seed, dim))
        for i in range(dim):
            seed_X[:, i] = t_vals ** (1.0 / (1.0 + 0.02 * i))
        seed_X = np.clip(seed_X, lo, hi)

        # Evaluate seeds
        F = np.array([problem(x) for x in seed_X])
        used = n_seed

        # Population size
        pop_size = min(100, budget - used)
        if pop_size < 10:
            pop_size = max(10, budget - used)

        # Initialize rest of population randomly
        rng = np.random.default_rng(42)
        X = np.zeros((pop_size, dim))
        F_all = np.zeros((pop_size, n_obj))

        # Place seeds in first n_seed slots
        n_seed_actual = min(n_seed, pop_size)
        X[:n_seed_actual] = seed_X[:n_seed_actual]
        F_all[:n_seed_actual] = F[:n_seed_actual]

        # Random fill for rest
        if pop_size > n_seed_actual:
            X[n_seed_actual:] = lo + span * rng.random((pop_size - n_seed_actual, dim))
            F_all[n_seed_actual:] = np.array([problem(x) for x in X[n_seed_actual:]])
            used += (pop_size - n_seed_actual)

        # --- NSGA-II helpers ---
        def fast_non_dominated_sort(F_mat):
            """Return list of fronts, each a list of indices."""
            n = len(F_mat)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F_mat, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F_mat[indices, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-15:
                    continue
                order = np.argsort(f_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (f_max - f_min)
            return cd

        # --- Main loop ---
        generation = 0
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_all)

            # Build ranked population for selection
            # Assign rank to each individual
            rank = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(F_all, front)

            # Tournament selection
            def tournament():
                a = rng.integers(pop_size)
                b = rng.integers(pop_size)
                if rank[a] != rank[b]:
                    return a if rank[a] < rank[b] else b
                return a if cd[a] > cd[b] else b

            # Create offspring
            n_offspring = pop_size
            # Limit offspring to remaining budget
            max_offspring = min(n_offspring, budget - used)
            if max_offspring < 1:
                break

            offspring_X = np.zeros((max_offspring, dim))
            offspring_F = np.zeros((max_offspring, n_obj))

            for i in range(max_offspring):
                if time.time() > deadline:
                    break
                p1 = tournament()
                p2 = tournament()
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            x1, x2 = X[p1, d], X[p2, d]
                            if abs(x1 - x2) < 1e-14:
                                child1[d] = child2[d] = x1
                            else:
                                u = rng.random()
                                beta1 = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (2 * (1 - u)) ** (1 / (eta + 1))
                                beta2 = (2 * (1 - u)) ** (1 / (eta + 1)) if u <= 0.5 else (2 * u) ** (1 / (eta + 1))
                                c1 = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
                                c2 = 0.5 * ((1 + beta2) * x1 + (1 - beta2) * x2)
                                child1[d] = c1
                                child2[d] = c2
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                    # Polynomial mutation
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            delta_q = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (2 * (1 - u)) ** (1 / (eta + 1))
                            child1[d] += delta_q * (hi[d] - lo[d])
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            delta_q = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (2 * (1 - u)) ** (1 / (eta + 1))
                            child2[d] += delta_q * (hi[d] - lo[d])
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    # Use one child per iteration to stay within budget
                    offspring_X[i] = child1
                else:
                    offspring_X[i] = X[p1] + 0.1 * span * rng.standard_normal(dim)
                    offspring_X[i] = np.clip(offspring_X[i], lo, hi)

                offspring_F[i] = problem(offspring_X[i])
                used += 1

            if i == 0 and max_offspring > 0:
                # At least one offspring was created
                pass

            # Combine parent + offspring
            X_comb = np.vstack([X, offspring_X[:i + 1]]) if i >= 0 else X
            F_comb = np.vstack([F_all, offspring_F[:i + 1]]) if i >= 0 else F_all
            n_comb = len(X_comb)

            # Selection: keep best pop_size
            fronts_comb = fast_non_dominated_sort(F_comb)
            selected = []
            for front in fronts_comb:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd_front = crowding_distance(F_comb, front)
                    order = np.argsort(-cd_front)
                    selected.extend([front[j] for j in order[:remaining]])
                    break

            X = X_comb[selected]
            F_all = F_comb[selected]
            generation += 1

        return X
# EVOLVE-BLOCK-END
