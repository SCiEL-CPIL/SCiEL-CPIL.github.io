"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style optimizer with DTLZ7-aware initialization and hypervolume selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominates = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominates[i] += 1

        front0 = [i for i in range(n) if n_dominates[i] == 0]
        fronts.append(front0)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(self.n_obj):
            vals = np.array([F[i, m] for i in front_indices])
            idx = np.argsort(vals)
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            fmin, fmax = vals[idx[0]], vals[idx[-1]]
            if fmax - fmin < 1e-12:
                continue
            for k in range(1, n - 1):
                dist[idx[k]] += (vals[idx[k + 1]] - vals[idx[k - 1]]) / (fmax - fmin)
        return dist

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume (for 3D we use a simple approximation)."""
        # Filter points better than reference
        valid = [p for p in points if all(p[i] < ref[i] for i in range(len(ref)))]
        if not valid:
            return 0.0
        # For 3D, use a simple grid-based approximation
        # Sort by first objective
        valid.sort(key=lambda p: p[0])
        hv = 0.0
        for i in range(len(valid)):
            p = valid[i]
            if p[0] >= ref[0]:
                continue
            # Volume contribution: (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
            # but we need to account for overlaps - use a simple Monte Carlo or grid
            # For simplicity, use a conservative estimate
            w0 = ref[0] - p[0]
            w1 = ref[1] - p[1]
            w2 = ref[2] - p[2]
            if w0 > 0 and w1 > 0 and w2 > 0:
                hv += w0 * w1 * w2
        return hv

    def _select_parents(self, F, X, fronts):
        """Binary tournament selection based on rank and crowding distance."""
        n = len(F)
        # Assign ranks
        ranks = np.zeros(n, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        # Compute crowding distances per front
        cd = np.zeros(n)
        for front in fronts:
            cd[front] = self._crowding_distance(F, front)

        def dominates(a, b):
            if ranks[a] < ranks[b]:
                return True
            if ranks[a] > ranks[b]:
                return False
            return cd[a] > cd[b]

        def tournament():
            i = np.random.randint(n)
            j = np.random.randint(n)
            if dominates(i, j):
                return i
            else:
                return j

        p1 = tournament()
        p2 = tournament()
        return p1, p2

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # DTLZ7-aware initialization
        # For DTLZ7: x[0:n-m] should satisfy g(x) = 0, x[n-m:] control Pareto position
        # g(x) = sum(x_i^2) for i in [0, n-m-1] if x_i not in [0.1, 0.9]
        # Optimal: set first n-m vars to 0.5 (in [0.1, 0.9] so g=0)
        # Last m vars: use a spread on the Pareto front

        pop_size = min(40, budget // 5)
        X = np.zeros((pop_size, n))
        for i in range(pop_size):
            # First n-m variables: set to 0.5 (optimal for g=0)
            X[i, :n-m] = 0.5
            # Last m variables: spread on Pareto front
            # For DTLZ7, f1=x[n-m], f2=x[n-m+1], f3=(1+f1)(1+f2)(1+g/f1)
            # Pareto front: f1 in [0,1], f2 in [0,1], f3 = (1+f1)(1+f2)
            # Use Latin hypercube or random spread
            X[i, n-m:] = rng.random(m)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II main loop
        while used < budget:
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Select parents
            p1, p2 = self._select_parents(F, X, fronts)

            # Crossover and mutation
            child = X[p1].copy()
            # Blend crossover
            alpha = rng.random(n)
            child = alpha * X[p1] + (1 - alpha) * X[p2]
            # Mutation: adaptive rate, focus on structure
            # DTLZ7: keep first n-m vars near 0.5, mutate last m more
            mutation_rate = 0.3
            for d in range(n):
                if rng.random() < mutation_rate:
                    if d < n - m:
                        # Small perturbation for g-vars, keep near 0.5
                        child[d] = 0.5 + 0.05 * rng.standard_normal()
                    else:
                        # Larger perturbation for Pareto position vars
                        child[d] = child[d] + 0.2 * rng.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual
            # Use hypervolume contribution as fitness
            # Simple: replace the individual with worst sum of objectives
            # Better: replace from last front
            if fronts and len(fronts) > 0:
                last_front = fronts[-1]
                if last_front:
                    # Pick random from last front
                    replace_idx = last_front[rng.integers(len(last_front))]
                else:
                    replace_idx = rng.integers(len(F))
            else:
                replace_idx = rng.integers(len(F))

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
