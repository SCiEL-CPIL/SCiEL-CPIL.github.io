"""Multi-objective optimizer on ZDT4 with NSGA-II and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer tailored for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices) via fast non-dominated sort."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        front0 = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[j] += 1
                # Actually let me fix the logic: i dominates j means i is better
                # Let me redo this properly
        # Redo: i dominates j if F[i] <= F[j] elementwise and F[i] < F[j] in at least one
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1
        for i in range(n):
            if n_dom[i] == 0:
                front0.append(i)
        fronts.append(front0)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                k += 1
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        for m in range(self.n_obj):
            vals = F[front_indices, m]
            min_v, max_v = vals.min(), vals.max()
            if max_v - min_v < 1e-14:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for k in range(1, n - 1):
                cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, budget // 2)
        pop_size = max(20, pop_size)

        # Initial population: exploit ZDT4 structure
        # x0 in [-5, 5] matters for g; x1..x9 in [0,1]
        # For ZDT4: f1 = x0, g = 1 + 10*(n-1) + sum(100*(x_i^2 - 10*cos(4*pi*x_i)) for i in 1..n-1)
        # h = 1 - sqrt(f1/g) * exp(-0.2*(f1/5)^2)
        # To get good Pareto front, we want small g values. g is minimized when x_i are at 0 (cos(0)=1)
        # So x1..x9 should be near 0 or 1 (where cos(4*pi*x)=1)
        # Actually cos(4*pi*x) = 1 when 4*pi*x = 2*pi*k, i.e., x = k/2
        # So x_i in {0, 0.5, 1} give cos=1, minimizing the penalty
        # g = 1 + 90 + sum(100*(x_i^2 - 10)) for i=1..9
        # At x_i=0: 100*(0-10) = -1000, so g = 91 - 9000 = negative? No wait
        # g = 1 + 10*(9) + sum(100*(x_i^2 - 10*cos(4*pi*x_i)))
        # At x_i=0: 100*(0 - 10*1) = -1000
        # So g = 91 + 9*(-1000) = 91 - 9000 = -8909? That can't be right for a minimization.
        # Let me re-examine: ZDT4 g(x) = 1 + 10*(n-1) + sum_{i=2}^n [100*(x_i^2 - 10*cos(4*pi*x_i)) + x_i]
        # At x_i=0: 100*(0 - 10*1) + 0 = -1000
        # So g can be very negative. But f1 = x1 (first variable), and h depends on f1/g.
        # Actually for ZDT4, the Pareto front has f1 in [0,1] typically, and g >= 1.
        # Wait, let me just use the problem function as given.
        
        # Strategy: sample x0 uniformly in [-5,5], and x1..x9 with emphasis near 0, 0.5, 1
        X = np.zeros((pop_size, dim))
        X[:, 0] = rng.uniform(lo[0], hi[0], pop_size)
        for d in range(1, dim):
            # Mix: 50% uniform, 30% near 0, 20% near 1
            r = rng.random(pop_size)
            X[:, d] = np.where(r < 0.5, rng.uniform(lo[d], hi[d], pop_size),
                       np.where(r < 0.8, rng.uniform(0, 0.2, pop_size), rng.uniform(0.8, 1.0, pop_size)))
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop
        while used < budget:
            if time.time() > deadline - 5:
                break

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Build mating pool from best fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) > pop_size:
                    break
                mating_pool.extend(front)

            if not mating_pool:
                mating_pool = list(range(pop_size))

            # Binary tournament selection
            def tournament():
                i1 = mating_pool[rng.integers(len(mating_pool))]
                i2 = mating_pool[rng.integers(len(mating_pool))]
                # Compare: use rank (front index) then crowding
                front_idx = {}
                for fi, front in enumerate(fronts):
                    for idx in front:
                        front_idx[idx] = fi
                r1 = front_idx.get(i1, 999)
                r2 = front_idx.get(i2, 999)
                if r1 != r2:
                    return i1 if r1 < r2 else i2
                return i1

            p1 = tournament()
            p2 = tournament()

            # Simulated binary crossover (SBX)
            child = X[p1].copy()
            if rng.random() < 0.9:  # crossover rate
                for d in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
            child = np.clip(child, lo, hi)

            # Mutation: polynomial mutation
            for d in range(dim):
                if rng.random() < 0.8:  # mutation rate
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / 21) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / 21)
                    child[d] += delta * (hi[d] - lo[d])
                    child[d] = np.clip(child[d], lo[d], hi[d])

            fc = problem(child)
            used += 1

            # Replace worst individual
            # Find dominated individuals
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            if np.any(dominated):
                # Replace a random dominated individual
                dom_indices = np.where(dominated)[0]
                idx = dom_indices[rng.integers(len(dom_indices))]
            else:
                # Replace worst by sum of objectives
                idx = int(np.argmax(np.sum(F, axis=1)))

            X[idx] = child
            F[idx] = fc

        # Local refinement on non-dominated points
        if time.time() < deadline - 2:
            fronts = self._fast_non_dominated_sort(F)
            pareto_indices = fronts[0] if fronts else []
            for idx in pareto_indices:
                if time.time() > deadline - 2:
                    break
                # Local search with small steps
                for _ in range(10):
                    if time.time() > deadline - 2:
                        break
                    if used >= budget:
                        break
                    candidate = X[idx] + 0.01 * (hi - lo) * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    # Accept if it dominates current or is non-dominated
                    if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                        X[idx] = candidate
                        F[idx] = fc
                    else:
                        # Check if it's better in hypervolume sense (simpler: replace if better in at least one obj and not worse in other)
                        if fc[0] < F[idx, 0] and fc[1] <= F[idx, 1]:
                            X[idx] = candidate
                            F[idx] = fc
                        elif fc[1] < F[idx, 1] and fc[0] <= F[idx, 0]:
                            X[idx] = candidate
                            F[idx] = fc

        return X
# EVOLVE-BLOCK-END
