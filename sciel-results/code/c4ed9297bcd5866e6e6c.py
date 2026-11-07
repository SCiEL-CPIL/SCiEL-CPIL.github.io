"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))

        # Random initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a set of front indices."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            # Map global indices to local positions
            idx_to_pos = {front_indices[i]: i for i in range(n)}

            for m in range(n_obj):
                sorted_local = sorted(range(n), key=lambda i: F[front_indices[i], m])
                dist[sorted_local[0]] = np.inf
                dist[sorted_local[-1]] = np.inf
                f_min = F[front_indices[sorted_local[0]], m]
                f_max = F[front_indices[sorted_local[-1]], m]
                if f_max - f_min < 1e-15:
                    continue
                for i in range(1, n - 1):
                    prev_idx = front_indices[sorted_local[i - 1]]
                    next_idx = front_indices[sorted_local[i + 1]]
                    dist[sorted_local[i]] += (F[next_idx, m] - F[prev_idx, m]) / (f_max - f_min)

            return dist

        def binary_tournament(F, fronts, crowd):
            """Select one individual via binary tournament."""
            i, j = rng.integers(0, len(F), size=2)
            fi = fronts[i]
            fj = fronts[j]
            if fi != fj:
                return i if fi < fj else j
            return i if crowd[i] > crowd[j] else j

        def sbx_crossover(x1, x2, lo, hi, dim):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                c1 = np.empty(dim)
                c2 = np.empty(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        c1[d] = x1[d]
                        c2[d] = x2[d]
                    else:
                        x1v, x2v = x1[d], x2[d]
                        lo_v, hi_v = lo[d], hi[d]
                        if abs(x1v - x2v) < 1e-14:
                            c1[d] = x1v
                            c2[d] = x2v
                            continue
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 21)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        c1[d] = 0.5 * ((1 + beta) * x1v + (1 - beta) * x2v)
                        c2[d] = 0.5 * ((1 - beta) * x1v + (1 + beta) * x2v)
                        c1[d] = np.clip(c1[d], lo_v, hi_v)
                        c2[d] = np.clip(c2[d], lo_v, hi_v)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x, lo, hi, dim, eta=20):
            """Polynomial mutation."""
            x = x.copy()
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    x1v, x2v = lo[d], hi[d]
                    if abs(x1v - x2v) < 1e-14:
                        continue
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x[d] = x[d] + delta_q * (x2v - x1v)
                    x[d] = np.clip(x[d], x1v, x2v)
            return x

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Crowding distance per front
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring // 2):
                p1 = binary_tournament(F, np.array([len(fronts) - 1 - fronts.index([i]) if i in [j for j in front] else 0 for i in range(len(F))]) if False else np.zeros(len(F), dtype=int), crowd)

                # Simpler: assign front rank
                front_rank = np.zeros(len(F), dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = rank

                # Tournament based on front rank and crowding
                def tournament_select():
                    i, j = rng.integers(0, len(F), size=2)
                    if front_rank[i] != front_rank[j]:
                        return i if front_rank[i] < front_rank[j] else j
                    return i if crowd[i] > crowd[j] else j

                p1 = tournament_select()
                p2 = tournament_select()

                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi, dim)
                c1 = polynomial_mutation(c1, lo, hi, dim)
                c2 = polynomial_mutation(c2, lo, hi, dim)
                offspring_X.append(c1)
                offspring_X.append(c2)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += len(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            next_front_rank = np.zeros(len(combined_F), dtype=int)
            for rank, front in enumerate(combined_fronts):
                for idx in front:
                    next_front_rank[idx] = rank

            # Crowding distance for combined
            combined_crowd = np.zeros(len(combined_F))
            for front in combined_fronts:
                if len(front) > 0:
                    cd = crowding_distance(combined_F, front)
                    for i, idx in enumerate(front):
                        combined_crowd[idx] = cd[i]

            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: combined_crowd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
