"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars).

Strategy: an NSGA-II-style evolutionary search, but with a strong structural
bias toward the known front geometry.  For WFG7 the Pareto front lies on a
unit hypersphere in the objective space (approximately), so we:
  1. seed a fraction of the initial population by sampling directions on the
     positive unit sphere and mapping them into decision space via a linear
     projection tuned to the problem;
  2. use fast non-dominated sorting + crowding distance for selection;
  3. apply a two-point crossover with a Gaussian mutation whose scale adapts
     to progress;
  4. enforce an explicit wall-clock deadline so we always return within the
     330 s budget.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300  # safety margin before 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (lo + hi)

        rng = np.random.default_rng(42)

        # ---- population size -------------------------------------------------
        pop_size = 50
        n_front_seeds = 20
        n_random = pop_size - n_front_seeds

        # ---- helpers ----------------------------------------------------------
        def evaluate(X):
            """Evaluate all rows of X; returns (n, n_obj) array."""
            F = np.empty((len(X), n_obj))
            for i in range(len(X)):
                F[i] = problem(X[i])
            return F

        def non_dominated_sort(F):
            """Return list of fronts (lists of indices) and an array of ranks."""
            n = len(F)
            domination = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        domination[i, j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        domination[j, i] = True
            fronts = []
            remaining = list(range(n))
            while remaining:
                front = [i for i in remaining
                         if not np.any(domination[[j for j in remaining if j != i], i])]
                if not front:
                    front = remaining
                fronts.append(front)
                remaining = [i for i in remaining if i not in front]
            ranks = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            return fronts, ranks

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        def two_point_crossover(p1, p2):
            a, b = sorted(rng.choice(dim, size=2, replace=False))
            child = p1.copy()
            child[a:b] = p2[a:b]
            return child

        def mutate(x, scale):
            return np.clip(x + scale * span * rng.standard_normal(dim), lo, hi)

        # ---- front seeding ----------------------------------------------------
        def front_seed():
            """Sample a direction on the positive unit sphere and map to decision space."""
            v = rng.standard_normal(n_obj)
            v = np.abs(v)
            v /= np.linalg.norm(v)
            # Map the first n_obj components of decision space to the front direction
            x = mid.copy()
            for k in range(min(n_obj, dim)):
                x[k] = lo[k] + v[k] * span[k]
            # Remaining dims at mid (front-neutral)
            return x

        # ---- initialize population ---------------------------------------------
        X = np.empty((pop_size, dim))
        for i in range(n_front_seeds):
            X[i] = front_seed()
        for i in range(n_front_seeds, pop_size):
            X[i] = lo + span * rng.random(dim)
        X = np.clip(X, lo, hi)

        F = evaluate(X)
        used = pop_size

        # ---- main loop ----------------------------------------------------------
        while used < budget:
            if time.time() > deadline:
                break

            fronts, ranks = non_dominated_sort(F)
            crowd = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # Build offspring
            remaining = budget - used
            n_off = min(pop_size, remaining)
            offX = np.empty((n_off, dim))
            offF = np.empty((n_off, n_obj))

            progress = used / max(1, budget)
            mut_scale = 0.12 * (1.0 - 0.60 * progress)

            for c in range(n_off):
                if time.time() > deadline:
                    break
                # Tournament selection (size 2)
                i1 = rng.integers(0, len(F))
                i2 = rng.integers(0, len(F))
                p1 = i1 if (ranks[i1] < ranks[i2] or
                            (ranks[i1] == ranks[i2] and crowd[i1] >= crowd[i2])) else i2
                j1 = rng.integers(0, len(F))
                j2 = rng.integers(0, len(F))
                p2 = j1 if (ranks[j1] < ranks[j2] or
                            (ranks[j1] == ranks[j2] and crowd[j1] >= crowd[j2])) else j2

                child = two_point_crossover(X[p1], X[p2])
                child = mutate(child, mut_scale)
                offX[c] = child
                offF[c] = problem(child)
                used += 1

            if used >= budget:
                break

            # Elitist replacement
            popX = np.vstack([X, offX])
            popF = np.vstack([F, offF])
            fronts, ranks = non_dominated_sort(popF)
            crowd = np.zeros(len(popF))
            for front in fronts:
                cd = crowding_distance(popF, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]
            order = np.lexsort((-crowd, ranks))
            keep = order[:pop_size]
            X = popX[keep]
            F = popF[keep]

        return X
