# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin from 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.RandomState(42)

        N = 20
        evals = 0

        # Initial population
        X = lo + (hi - lo) * rng.rand(N, self.dim)
        F = np.array([problem(x) for x in X], dtype=float)
        evals = N

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
            current_front = [i for i in range(n) if nd[i] == 0]
            if not current_front:
                return [list(range(n))]
            front_idx = 0
            fronts.append(current_front)
            while current_front:
                next_front = []
                for i in current_front:
                    rank[i] = front_idx
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = front_idx + 1
                            next_front.append(j)
                front_idx += 1
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range
            return cd

        def tournament_select(ranks, F, n_sel=2):
            idx = rng.randint(len(ranks), size=n_sel)
            best = idx[0]
            for i in range(1, n_sel):
                if ranks[idx[i]] < ranks[best]:
                    best = idx[i]
            return best

        def sbx_crossover(x1, x2):
            if rng.rand() < 0.5:
                return x1.copy()
            eta = 20.0
            u = rng.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            return c1, c2

        def polynomial_mutation(x, eta=20.0):
            y = x.copy()
            for i in range(len(x)):
                if rng.rand() < 1.0 / self.dim:
                    u = rng.rand()
                    d = hi[i] - lo[i]
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    y[i] += delta * d
            return np.clip(y, lo, hi)

        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            children_X = []
            for _ in range(N):
                # Selection
                parents = [tournament_select(ranks, F) for _ in range(2)]
                x1, x2 = X[parents[0]], X[parents[1]]

                # Crossover
                if rng.rand() < 0.9:
                    c1, c2 = sbx_crossover(x1, x2)
                else:
                    c1, c2 = x1.copy(), x2.copy()

                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                if evals < self.budget:
                    children_X.append(c1)
                    evals += 1
                if len(children_X) < N and evals < self.budget:
                    children_X.append(c2)
                    evals += 1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in children_X], dtype=float)

            # Combine
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort
            fronts = non_dominated_sort(F_all)
            ranks = np.zeros(len(F_all), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Select next population
            next_X = []
            next_F = []
            for rank, front in enumerate(fronts):
                if len(next_X) + len(front) <= N:
                    next_X.extend(X_all[front])
                    next_F.extend(F_all[front])
                else:
                    # Sort by crowding distance
                    cd = crowding_distance(F_all, front)
                    front_idx = np.array(front)
                    sorted_front = front_idx[np.argsort(-cd)]
                    need = N - len(next_X)
                    next_X.extend(X_all[sorted_front[:need]])
                    next_F.extend(F_all[sorted_front[:need]])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
