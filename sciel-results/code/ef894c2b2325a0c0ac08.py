"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a good fraction of budget
        pop_size = 20
        if budget < 40:
            pop_size = max(4, budget // 2)

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (all objectives <=, at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            fronts = []
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-12:
                    continue
                # Sort by objective m
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)
            return cd

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated Binary Crossover."""
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for d in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        child1[d] = x1[d]
                        child2[d] = x2[d]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        c2 = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                        child1[d] = c1
                        child2[d] = c2
                else:
                    child1[d] = x2[d]
                    child2[d] = x1[d]
            return child1, child2

        def mutate(x, sigma=0.1, eta=20.0):
            """Polynomial mutation."""
            child = x.copy()
            for d in range(dim):
                if np.random.rand() < 0.5:
                    lo_d, hi_d = lo[d], hi[d]
                    if hi_d - lo_d < 1e-14:
                        continue
                    u = np.random.rand()
                    if u < 0.5:
                        val = 2 * u
                        beta = (1 + (eta + 1) * val) ** (-1.0 / (eta + 1.0))
                    else:
                        val = 2 * (1 - u)
                        beta = (1 + (eta + 1) * val) ** (-1.0 / (eta + 1.0))
                    delta = beta * (hi_d - lo_d)
                    if u < 0.5:
                        child[d] = x[d] - delta
                    else:
                        child[d] = x[d] + delta
                    child[d] = np.clip(child[d], lo_d, hi_d)
            return child

        # Main loop
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break

                # Tournament selection from current population
                fronts = non_dominated_sort(F)
                # Assign ranks
                rank = np.zeros(len(F), dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = r

                # Select parents via tournament
                def tournament_select():
                    p1 = np.random.randint(len(F))
                    p2 = np.random.randint(len(F))
                    if rank[p1] < rank[p2]:
                        return p1
                    elif rank[p2] < rank[p1]:
                        return p2
                    # Same rank, use crowding distance
                    cd = crowding_distance(F, list(range(len(F))))
                    if cd[p1] > cd[p2]:
                        return p1
                    else:
                        return p2

                p1 = tournament_select()
                p2 = tournament_select()

                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                # Mutation
                c1 = mutate(c1, sigma=0.1)
                c2 = mutate(c2, sigma=0.1)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = non_dominated_sort(F_all)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    # Need to truncate this front by crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_all, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    for i in range(min(remaining, len(order))):
                        idx = front[order[i]]
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)
            gen += 1

        return X
# EVOLVE-BLOCK-END
