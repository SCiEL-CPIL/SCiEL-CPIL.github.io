"""Rewritten optimizer for WFG4 multi-objective problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using a hybrid strategy:
    1. Latin Hypercube Sampling (LHS) for broad initial coverage.
    2. Informed local search around the best non-dominated points found.
    3. Time-aware budget management to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # We will record all evaluated points (x, f) in lists
        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            f = np.array(problem(x), dtype=np.float64)
            X_list.append(x.copy())
            F_list.append(f.copy())
            eval_count += 1
            return f

        def is_dominated(f1, f2):
            """Return True if f1 dominates f2 (f1 <= f2 and f1 < f2 in at least one)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_nondominated_sort(F):
            """
            Returns list of fronts, each a list of indices.
            """
            m = len(F)
            S = [[] for _ in range(m)]  # S[i] = set of indices dominated by i
            n_d = np.zeros(m, dtype=int)  # number of points dominating i
            rank = np.zeros(m, dtype=int)
            fronts = []

            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if is_dominated(F[i], F[j]):
                        S[i].append(j)
                    elif is_dominated(F[j], F[i]):
                        n_d[i] += 1
            for i in range(m):
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            k = 0
            while fronts[k]:
                Q = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            Q.append(j)
                k += 1
                if Q:
                    fronts.append(Q)
            return fronts

        # --- Phase 1: Latin Hypercube Sampling for broad coverage ---
        n_lhs = 100  # Use half the budget for LHS
        n_lhs = min(n_lhs, self.budget)
        # Generate LHS points
        # For each dimension, divide [0,1] into n_lhs equal intervals,
        # pick one point in each interval, and permute across dimensions.
        u = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            u[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        # Map to [lo, hi]
        X_lhs = lo + u * (hi - lo)
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            evaluate(X_lhs[i])

        # --- Phase 2: Informed local search around best non-dominated points ---
        # Find the current non-dominated set
        if F_list:
            F_arr = np.array(F_list)
            fronts = fast_nondominated_sort(F_arr)
            # Take the first front (best non-dominated)
            if fronts:
                best_indices = fronts[0]
                # For each best point, do a few local perturbations
                n_local = (self.budget - eval_count) // max(len(best_indices), 1)
                n_local = max(1, min(n_local, 10))  # Cap local steps per parent
                for idx in best_indices:
                    if eval_count >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    parent_x = X_list[idx]
                    # Try a few perturbations with decreasing step size
                    for step_scale in [0.1, 0.05, 0.02]:
                        if eval_count >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        step = step_scale * (hi - lo)
                        child_x = parent_x + step * rng.standard_normal(self.dim)
                        child_x = np.clip(child_x, lo, hi)
                        evaluate(child_x)

        # --- Phase 3: Fill remaining budget with random sampling ---
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + rng.random(self.dim) * (hi - lo)
            evaluate(x)

        return X_list
# EVOLVE-BLOCK-END
