"""Multi-objective optimizer on DTLZ7 using NSGA-II with power-law biased initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with hypervolume-aware initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a moderate population for good diversity
        pop_size = min(50, max(10, self.budget // 4))

        # Power-law biased initialization for DTLZ7
        # DTLZ7 Pareto front: f1 = x1, f2 = x2, f3 = (1+g)*h
        # where h = n_obj - sum_{i=1}^{n_obj-1} (f_i/(1+g)) * (1 + sin(pi*f_i/(1+g)))
        # For hypervolume maximization with reference point [1.1, 1.1, 25.0],
        # we want points with low f1, low f2, and low f3.
        # The Pareto front has f1, f2 in [0,1] and f3 depends on g.
        # Bias towards low f1 and f2 using power law.

        def make_initial_population(n):
            """Create initial population with power-law bias towards hypervolume-optimal region."""
            X = np.zeros((n, self.dim))
            for i in range(n):
                # DTLZ7: first dim-1 variables determine the objective space position
                # Use power-law distribution biased towards 0 (low f1, f2)
                # f1 = x1, f2 = x2 for DTLZ7
                # Power law: x = u^(1/alpha) biases towards 0 when alpha > 1
                alpha = 2.0  # stronger bias towards origin
                x = np.ones(self.dim)
                for j in range(self.dim - 1):
                    x[j] = rng.random() ** (1.0 / alpha)
                # Last variable affects g; set it to maximize g contribution
                # g = 100 * (n_g + sum((x_j - 0.5)^2 - cos(20*pi*(x_j - 0.5))) for j in n_g)
                # Minimum g is 0 when x_j = 0.5 for all g variables
                x[-1] = 0.5
                X[i] = x
            return X

        # Initialize population
        X = make_initial_population(pop_size)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Check deadline
        if time.time() > deadline:
            return X

        # NSGA-II main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Crowding distance assignment
            for front in fronts:
                if len(front) > 0:
                    self._assign_crowding_distance(F, front)

            # Selection: tournament selection from non-dominated fronts
            parents = []
            for _ in range(2):
                # Tournament selection
                idx1 = rng.integers(len(F))
                idx2 = rng.integers(len(F))
                # Compare by front rank, then crowding distance
                rank1 = fronts.index([i for i in range(len(F)) if i in set(fronts[0])] if False else None)
                # Simpler: use front assignment
                parent_idx = self._tournament_select(F, fronts, rng)
                parents.append(X[parent_idx])

            # Crossover (SBX-like)
            child = self._crossover(parents[0], parents[1], lo, hi, rng)

            # Mutation
            child = self._mutate(child, lo, hi, rng)

            # Evaluate
            fc = problem(child)
            used += 1

            # Environmental selection: combine parent and offspring populations
            # For simplicity, replace worst point in current population
            # Use hypervolume contribution as a proxy for quality
            worst_idx = self._find_worst_index(F)
            X[worst_idx] = child
            F[worst_idx] = fc

        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting."""
        n = len(F)
        fronts = []
        S = [[] for _ in range(n)]  # S[i] = list of j dominated by i
        n_dominance = np.zeros(n, dtype=int)  # number of points dominating i
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominance[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominance[i] += 1

        # First front
        front = [i for i in range(n) if n_dominance[i] == 0]
        for i in front:
            rank[i] = 0
        if front:
            fronts.append(front)

        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominance[j] -= 1
                    if n_dominance[j] == 0:
                        next_front.append(j)
                        rank[j] = k + 1
            k += 1
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _assign_crowding_distance(self, F, front):
        """Assign crowding distance to points in a front."""
        if len(front) <= 2:
            for i in front:
                F[i, -1] = np.inf  # Store crowding distance in last column temporarily
            return

        crowding = np.zeros(len(F))
        for m in range(self.n_obj):
            # Sort front by objective m
            sorted_idx = sorted(front, key=lambda i: F[i, m])
            crowding[sorted_idx[0]] = np.inf
            crowding[sorted_idx[-1]] = np.inf
            f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
            if f_range > 1e-10:
                for i in range(1, len(sorted_idx) - 1):
                    crowding[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / f_range

        # Store crowding distance temporarily
        for i in front:
            F[i, -1] = crowding[i]

    def _tournament_select(self, F, fronts, rng):
        """Tournament selection based on front rank and crowding distance."""
        # Assign front ranks
        n = len(F)
        rank = np.zeros(n, dtype=int)
        for k, front in enumerate(fronts):
            for i in front:
                rank[i] = k

        # Get crowding distances (stored in last column)
        crowding = F[:, -1].copy()

        # Tournament of size 2
        idx1 = rng.integers(n)
        idx2 = rng.integers(n)

        # Compare: lower rank is better, then higher crowding distance
        if rank[idx1] < rank[idx2]:
            return idx1
        elif rank[idx2] < rank[idx1]:
            return idx2
        else:
            # Same rank, use crowding distance
            if crowding[idx1] > crowding[idx2]:
                return idx1
            else:
                return idx2

    def _crossover(self, p1, p2, lo, hi, rng):
        """Simple blend crossover."""
        child = np.zeros(self.dim)
        for j in range(self.dim):
            if rng.random() < 0.5:
                child[j] = p1[j]
            else:
                child[j] = p2[j]
        # Add some blending
        alpha = rng.random()
        child = alpha * p1 + (1 - alpha) * p2
        return np.clip(child, lo, hi)

    def _mutate(self, x, lo, hi, rng):
        """Polynomial-like mutation."""
        child = x.copy()
        rate = 0.1
        for j in range(self.dim):
            if rng.random() < rate:
                # Gaussian mutation
                scale = 0.1 * (hi[j] - lo[j])
                child[j] += rng.standard_normal() * scale
        return np.clip(child, lo, hi)

    def _find_worst_index(self, F):
        """Find the worst point to replace using sum of normalized objectives."""
        # Normalize objectives
        F_norm = F.copy()
        for m in range(self.n_obj):
            f_min = np.min(F[:, m])
            f_max = np.max(F[:, m])
            if f_max - f_min > 1e-10:
                F_norm[:, m] = (F[:, m] - f_min) / (f_max - f_min)
            else:
                F_norm[:, m] = 0.0

        # Worst is the one with highest sum
        return int(np.argmax(np.sum(F_norm, axis=1)))
# EVOLVE-BLOCK-END
