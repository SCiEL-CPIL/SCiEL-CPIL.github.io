import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: aim for ~5 generations
        pop_size = min(40, budget // 5)
        if pop_size < 10:
            pop_size = max(10, budget // 4)

        # Initial population
        def rand_pop(n):
            return lo + (hi - lo) * np.random.rand(n, dim)

        X = rand_pop(pop_size)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            # a dominates b if a <= b in all and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = [0] * n  # number of solutions dominating i
            rank = [0] * n
            front = [0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            i = 0
            while front[i] > 0:
                i += 1
                next_front = []
                for k in range(front[i - 1]):
                    p = next_front  # placeholder
                # Re-do properly
            # Let me rewrite this cleanly
            return None  # placeholder, I'll rewrite

        # Let me write a clean fast non-dominated sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front.append(i)
            fronts = []
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = k + 1
                            next_front.append(q)
                front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, idx):
            n = len(idx)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(n_obj):
                vals = [F[i, m] for i in idx]
                vmin = min(vals)
                vmax = max(vals)
                if vmax == vmin:
                    continue
                # sort by this objective
                order = sorted(range(n), key=lambda i: F[idx[i], m])
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                for k in range(1, n - 1):
                    dist[order[k]] += (F[idx[order[k + 1]], m] - F[idx[order[k - 1]], m]) / (vmax - vmin)
            return dist

        def tournament_select(rank, crowd, F, n_sel):
            # n_sel: number of parents to select
            parents = []
            for _ in range(n_sel):
                i = np.random.randint(len(rank))
                j = np.random.randint(len(rank))
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[i] > rank[j]:
                    parents.append(j)
                else:
                    if crowd[i] >= crowd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            # Simulated Binary Crossover
            n = len(x1)
            c1 = np.zeros(n)
            c2 = np.zeros(n)
            for i in range(n):
                if np.random.rand() < 0.5:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                    continue
                x_lo = min(x1[i], x2[i])
                x_hi = max(x1[i], x2[i])
                if x_hi == x_lo:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                c1[i] = 0.5 * ((x_hi + x_lo) - beta * (x_hi - x_lo))
                c2[i] = 0.5 * ((x_hi + x_lo) + beta * (x_hi - x_lo))
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutate(x, lo, hi, eta=20.0):
            n = len(x)
            y = x.copy()
            for i in range(n):
                if np.random.rand() < 0.5:
                    continue
                if x[i] == lo[i] or x[i] == hi[i]:
                    continue
                u = np.random.rand()
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                if u <= 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * xy ** (eta + 1)
                    deltaq = val ** (1.0 / (eta + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * xy ** (eta + 1)
                    deltaq = 1 - val ** (1.0 / (eta + 1))
                y[i] = x[i] + deltaq * (hi[i] - lo[i])
                y[i] = np.clip(y[i], lo[i], hi[i])
            return y

        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring_X = []
            for _ in range(pop_size // 2):
                p1 = tournament_select(rank, crowd, F, 1)[0]
                p2 = tournament_select(rank, crowd, F, 1)[0]
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                c1 = polynomial_mutate(c1, lo, hi)
                c2 = polynomial_mutate(c2, lo, hi)
                offspring_X.append(c1)
                offspring_X.append(c2)

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort
            fronts, rank_all = fast_non_dominated_sort(F_all)

            # Select next population
            X_new = []
            F_new = []
            rank_new = []
            crowd_new = []
            remaining = pop_size
            for front in fronts:
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    # Take all
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                        rank_new.append(rank_all[idx])
                    remaining -= len(front)
                else:
                    # Need crowding distance
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    order = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[order[i]]
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                        rank_new.append(rank_all[idx])
                    remaining = 0

            X = np.array(X_new)
            F = np.array(F_new)
            rank = np.array(rank_new)
            # Recompute crowding distance for the new population
            cd_full = crowding_distance(F, list(range(len(F))))
            crowd = np.array(cd_full)

        return X
