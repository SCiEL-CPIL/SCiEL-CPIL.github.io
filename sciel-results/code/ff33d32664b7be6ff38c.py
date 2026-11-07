"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Use the global numpy random state (already seeded by evaluator)
        # Population size: start with 30, grow to 50
        pop_size = 30
        X = lo + range_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Hypervolume contribution helper
        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = False
                        break
            return nd

        def crowding_distance(F, nd_mask):
            """Compute crowding distances for non-dominated points."""
            n = len(F)
            dist = np.zeros(n)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) <= 2:
                dist[nd_indices] = np.inf
                return dist
            for k in range(self.n_obj):
                sorted_idx = nd_indices[np.argsort(F[nd_indices, k])]
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                fmin = F[sorted_idx[0], k]
                fmax = F[sorted_idx[-1], k]
                if fmax > fmin:
                    for i in range(1, len(sorted_idx) - 1):
                        dist[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / (fmax - fmin)
            return dist

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = non_dominated_mask(F)
            n_nd = np.sum(nd_mask)

            # Select parents via tournament on non-dominated
            if n_nd >= 2:
                # Pick 2 random indices from non-dominated
                nd_indices = np.where(nd_mask)[0]
                i = np.random.randint(0, len(nd_indices))
                j = np.random.randint(0, len(nd_indices))
                p1 = X[nd_indices[i]]
                p2 = X[nd_indices[j]]
            else:
                i = np.random.randint(0, len(X))
                j = np.random.randint(0, len(X))
                p1 = X[i]
                p2 = X[j]

            # SBX crossover
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.random() < 0.5:
                    # Blend
                    alpha = np.random.random()
                    child1[d] = alpha * p1[d] + (1 - alpha) * p2[d]
                    child2[d] = (1 - alpha) * p1[d] + alpha * p2[d]
                else:
                    child1[d] = p1[d]
                    child2[d] = p2[d]

            # Polynomial mutation with adaptive eta
            eta = 20.0
            for child in (child1, child2):
                for d in range(self.dim):
                    if np.random.random() < 0.5:
                        u = np.random.random()
                        if u <= 0.5:
                            delta_ = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta_ = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        child[d] += delta_ * (hi[d] - lo[d])

            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)

            # Evaluate children (check budget and deadline)
            if used + 1 <= self.budget and time.time() < deadline:
                f1 = problem(child1)
                used += 1
                # Replace worst point in F
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child1
                F[worst] = f1

            if used + 1 <= self.budget and time.time() < deadline:
                f2 = problem(child2)
                used += 1
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child2
                F[worst] = f2

            # Periodically add diversity via random reinsertion
            if used % 50 == 0 and used + 1 <= self.budget and time.time() < deadline:
                x_rand = lo + range_ * np.random.random(self.dim)
                f_rand = problem(x_rand)
                used += 1
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = x_rand
                F[worst] = f_rand

        return X
# EVOLVE-BLOCK-END
