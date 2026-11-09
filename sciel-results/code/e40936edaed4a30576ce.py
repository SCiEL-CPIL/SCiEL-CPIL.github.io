"""Combined multi-objective optimizer: DE mutation + non-dominated sorting + crowding distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Use global np.random state so evaluator's np.random.seed(seed) differentiates runs
        rng = np.random

        # Population size: keep it moderate to allow many generations within budget
        N = min(40, max(20, self.budget // 5))
        max_evals = self.budget

        # Initialize population
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N

        def non_dominated_sort(F):
            """Return list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Does j dominate i?
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1

            current_front = [i for i in range(n) if nd[i] == 0]
            if current_front:
                fronts.append(current_front)

            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                k += 1

            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to points in indices."""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            n_obj = F.shape[1]
            for obj in range(n_obj):
                vals = F[indices, obj]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_val = vals[order[-1]] - vals[order[0]]
                if rng_val > 1e-15:
                    for k in range(1, m - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_val
            return cd

        def select_parents(F, X, n_sel):
            """Select n_sel parent indices using non-dominated sorting + crowding distance."""
            fronts = non_dominated_sort(F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_sel:
                    selected.extend(front)
                else:
                    remaining = n_sel - len(selected)
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    selected.extend(front[order[:remaining]])
                    break
            # If we still need more (shouldn't happen), fill with random
            while len(selected) < n_sel:
                selected.append(rng.integers(N))
            return selected

        def dominates(a, b):
            """Check if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        # DE parameters
        CR = 0.9
        F_mut = 0.8

        # Main loop
        while used < max_evals:
            if time.time() > deadline:
                break

            # Select parents via non-dominated sorting
            parent_indices = select_parents(F, X, N)

            # Generate one offspring per parent slot, but stop if budget exhausted
            for i in range(N):
                if used >= max_evals or time.time() > deadline:
                    break

                # Find three distinct parents different from i
                candidates = [p for p in parent_indices if p != i]
                if len(candidates) < 3:
                    candidates = [p for p in range(N) if p != i]
                if len(candidates) < 3:
                    continue

                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # DE mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)

                # Binomial crossover
                trial = X[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Selection: tournament-style
                if dominates(f_trial, F[i]):
                    X[i] = trial
                    F[i] = f_trial
                elif not dominates(F[i], f_trial):
                    # Incomparable: accept with some probability to maintain diversity
                    if rng.random() < 0.4:
                        X[i] = trial
                        F[i] = f_trial

        # Final local polish: refine top non-dominated points with small steps
        if time.time() < deadline - 5.0:
            # Find non-dominated indices
            fronts = non_dominated_sort(F)
            nd_indices = fronts[0] if fronts else []
            # Polish each non-dominated point with a few random perturbations
            for idx in nd_indices:
                if used >= max_evals or time.time() > deadline - 2.0:
                    break
                # Try a few small perturbations
                for _ in range(3):
                    if used >= max_evals or time.time() > deadline - 2.0:
                        break
                    step = 0.05 * span
                    delta = (rng.random(self.dim) * 2 - 1) * step
                    x_trial = np.clip(X[idx] + delta, lo, hi)
                    f_trial = problem(x_trial)
                    used += 1
                    if dominates(f_trial, F[idx]):
                        X[idx] = x_trial
                        F[idx] = f_trial

        return X
# EVOLVE-BLOCK-END
