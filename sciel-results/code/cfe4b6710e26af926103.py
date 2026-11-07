"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = self.budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            f = F[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(f[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf

                f_range = f[sorted_idx[-1], m] - f[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (f[sorted_idx[k + 1], m] - f[sorted_idx[k - 1], m]) / f_range

            return dist

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Create mating pool: select parents from best fronts
            # Use tournament selection based on rank and crowding distance
            parent1 = self._tournament_select(fronts, F, rng)
            parent2 = self._tournament_select(fronts, F, rng)

            # SBX crossover
            child = self._sbx_crossover(parent1, parent2, lo, hi, rng)

            # Mutation
            if rng.random() < 0.5:
                child = self._polynomial_mutation(child, lo, hi, rng)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace worst individual
            # Find the worst based on rank, then crowding distance
            worst_idx = self._find_worst(F, fronts)
            X[worst_idx] = child
            F[worst_idx] = fc

        return X

    def _tournament_select(self, fronts, F, rng):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i

        # Compute crowding distance per front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 0:
                cd[front] = crowding_distance(F, front)

        # Tournament of size 2
        a = rng.integers(n)
        b = rng.integers(n)
        if rank[a] < rank[b]:
            return a
        elif rank[b] < rank[a]:
            return b
        else:
            return a if cd[a] > cd[b] else b

    def _sbx_crossover(self, p1, p2, lo, hi, rng):
        """Simulated binary crossover."""
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / 20)
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / 20)
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            child[i] = np.clip(child[i], lo[i], hi[i])
        return child

    def _polynomial_mutation(self, x, lo, hi, rng):
        """Polynomial mutation."""
        y = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / 20) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 20)
                y[i] += delta * (hi[i] - lo[i])
                y[i] = np.clip(y[i], lo[i], hi[i])
        return y

    def _find_worst(self, F, fronts):
        """Find the worst individual to replace."""
        # Get the last front (worst rank)
        if len(fronts) == 0:
            return 0
        worst_front = fronts[-1]
        if len(worst_front) == 1:
            return worst_front[0]

        # Within worst front, use crowding distance (replace least crowded)
        cd = crowding_distance(F, worst_front)
        worst_local = np.argmin(cd)
        return worst_front[worst_local]
# EVOLVE-BLOCK-END
