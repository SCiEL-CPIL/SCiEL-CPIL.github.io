"""Multi-objective optimizer on ZDT6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(100, self.budget // 2)
        pop_size = max(pop_size, 10)

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        evals = pop_size

        # Archive of all evaluated points (for hypervolume)
        archive_X = X.copy()
        archive_F = F.copy()

        # NSGA-II main loop
        while evals < self.budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Crowding distance
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 1:
                    crowd[front] = self._crowding_distance(F[front])

            # Selection: binary tournament
            parents = []
            while len(parents) < pop_size:
                i, j = rng.integers(0, len(F), size=2)
                fi = fronts.index(i) if i in [x for f in fronts for x in f] else 0
                fj = fronts.index(j) if j in [x for f in fronts for x in f] else 0
                # Find which front
                fi = self._find_front(fronts, i)
                fj = self._find_front(fronts, j)
                if fi < fj:
                    parents.append(i)
                elif fj < fi:
                    parents.append(j)
                else:
                    if crowd[i] >= crowd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)

            # Crossover and mutation
            children_X = []
            children_F = []
            for k in range(0, pop_size - 1, 2):
                p1, p2 = parents[k], parents[k + 1]
                # Simulated binary crossover
                if rng.random() < 0.9:
                    c1, c2 = self._sbx(X[p1], X[p2], span, 20.0)
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()
                # Mutation
                c1 = self._mutate(c1, lo, hi, span, 1.0 / self.dim)
                c2 = self._mutate(c2, lo, hi, span, 1.0 / self.dim)
                children_X.append(c1)
                children_X.append(c2)

            # Evaluate children
            for cx in children_X:
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                fc = problem(cx)
                children_F.append(np.array(fc, dtype=float))
                evals += 1

            if not children_F:
                break

            children_F = np.array(children_F)

            # Combine parents and children
            X_all = np.vstack([X, np.array(children_X[:len(children_F)])])
            F_all = np.vstack([F, children_F])

            # Update archive
            archive_X = np.vstack([archive_X, np.array(children_X[:len(children_F)])])
            archive_F = np.vstack([archive_F, children_F])

            # Selection: keep best pop_size
            fronts_all = self._fast_non_dominated_sort(F_all)
            crowd_all = np.zeros(len(F_all))
            for front in fronts_all:
                if len(front) > 1:
                    crowd_all[front] = self._crowding_distance(F_all[front])

            # Fill new population
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(F_all[front])
                    new_X.append(X_all[front])
                    new_F.append(F_all[front])
                else:
                    needed = pop_size - len(new_F)
                    # Sort by crowding distance
                    idx = np.argsort(-crowd_all[front])[:needed]
                    front_idx = np.array(front)[idx]
                    new_X.append(X_all[front_idx])
                    new_F.append(F_all[front_idx])
                    break

            X = np.vstack(new_X)
            F = np.vstack(new_F)

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[j] += 1

        current_front = []
        for i in range(n):
            if n_dom[i] == 0:
                current_front.append(i)

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        crowd = np.zeros(n)
        for m in range(F.shape[1]):
            idx = np.argsort(F[:, m])
            crowd[idx[0]] = np.inf
            crowd[idx[-1]] = np.inf
            for i in range(1, n - 1):
                if F[idx[i + 1], m] - F[idx[i - 1], m] > 0:
                    crowd[idx[i]] += (F[idx[i + 1], m] - F[idx[i - 1], m]) / (F[idx[i + 1], m] - F[idx[i - 1], m] + 1e-10)
        return crowd

    def _find_front(self, fronts, idx):
        for i, front in enumerate(fronts):
            if idx in front:
                return i
        return 0

    def _sbx(self, x1, x2, span, eta):
        c1, c2 = x1.copy(), x2.copy()
        for i in range(len(x1)):
            if x1[i] == x2[i]:
                continue
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
            c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
        c1 = np.clip(c1, x1 - span, x1 + span)
        c2 = np.clip(c2, x1 - span, x1 + span)
        return c1, c2

    def _mutate(self, x, lo, hi, span, prob):
        x = x.copy()
        mask = rng.random(len(x)) < prob
        for i in np.where(mask)[0]:
            u = rng.random()
            if u <= 0.5:
                delta = (2 * u) ** (1 / 21) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / 21)
            x[i] += delta * span[i]
        return np.clip(x, lo, hi)
# EVOLVE-BLOCK-END
