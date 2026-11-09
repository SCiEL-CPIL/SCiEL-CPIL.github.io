"""Multi-objective optimizer on WFG4 with refined NSGA-II + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style optimizer for WFG4 multi-objective optimization with LHS init and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, indices):
        """Compute crowding distance for points at given indices."""
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        n_obj = F.shape[1]
        for k in range(n_obj):
            vals = F[indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    def _lhs_sample(self, rng, n, dim, lo, hi):
        """Latin Hypercube Sampling for better initial coverage."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * samples

    def _local_polish(self, problem, bounds, x, lo, hi, rng, max_evals=10):
        """Local gradient-free polish around a point using adaptive step sizes."""
        dim = len(x)
        best_x = x.copy()
        best_f = np.array(problem(best_x))
        evals_used = 1
        step_size = 0.05

        while evals_used < max_evals:
            improved = False
            for d in range(dim):
                if evals_used >= max_evals:
                    break
                for sign in [1, -1]:
                    if evals_used >= max_evals:
                        break
                    trial = best_x.copy()
                    trial[d] += sign * step_size * (hi[d] - lo[d])
                    trial = np.clip(trial, lo, hi)
                    ft = problem(trial)
                    evals_used += 1
                    if np.all(ft <= best_f) and np.any(ft < best_f):
                        best_x = trial
                        best_f = ft
                        improved = True
                        break
                if improved:
                    break
            if improved:
                step_size *= 0.7
            else:
                step_size *= 0.5
                if step_size < 0.001:
                    break
        return best_x, best_f, evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Population size
        pop_size = 30

        # Reserve budget for local polish
        polish_budget = 10
        main_budget = budget - polish_budget

        # Initialize population using LHS for better coverage
        X = self._lhs_sample(rng, pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE parameters
        F_CR = 0.8
        CR = 0.9
        K = 0.5

        while used < main_budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Selection: pick parents from best front using tournament
            front0 = fronts[0]
            if len(front0) < 2:
                parent_pool = list(range(pop_size))
            else:
                parent_pool = front0

            # Pick 3 distinct parents for DE mutation
            if len(parent_pool) >= 3:
                p1, p2, p3 = rng.choice(parent_pool, size=3, replace=False)
            else:
                p1 = rng.choice(parent_pool)
                p2 = rng.choice(parent_pool)
                p3 = rng.choice(parent_pool)

            # DE mutation
            v = X[p1] + K * (X[p2] - X[p3])

            # DE crossover
            jrand = rng.integers(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[p1][j]

            trial = np.clip(trial, lo, hi)

            # Evaluate
            ft = problem(trial)
            used += 1

            # Decide replacement using non-domination
            dominated_by_trial = []
            for i in range(pop_size):
                if np.all(ft <= F[i]) and np.any(ft < F[i]):
                    dominated_by_trial.append(i)

            if dominated_by_trial:
                sums = np.sum(F[dominated_by_trial], axis=1)
                worst_idx = dominated_by_trial[np.argmax(sums)]
                X[worst_idx] = trial
                F[worst_idx] = ft
            else:
                dominates_trial = []
                for i in range(pop_size):
                    if np.all(F[i] <= ft) and np.any(F[i] < ft):
                        dominates_trial.append(i)

                if dominates_trial:
                    worst_front = fronts[-1]
                    if worst_front:
                        cd = self._crowding_distance(F, worst_front)
                        worst_pos = np.argmin(cd)
                        worst_idx = worst_front[worst_pos]
                        if np.sum(ft) < np.sum(F[worst_idx]):
                            X[worst_idx] = trial
                            F[worst_idx] = ft
                else:
                    worst_front = fronts[-1]
                    if worst_front and len(worst_front) > 1:
                        cd = self._crowding_distance(F, worst_front)
                        worst_pos = np.argmin(cd)
                        worst_idx = worst_front[worst_pos]
                        X[worst_idx] = trial
                        F[worst_idx] = ft
                    else:
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                        X[worst_idx] = trial
                        F[worst_idx] = ft

        # Final local polish on non-dominated points
        if used < budget and time.time() < deadline:
            fronts = self._fast_non_dominated_sort(F)
            front0 = fronts[0]
            # Polish top non-dominated points
            n_polish = min(len(front0), max(3, polish_budget // 10))
            # Pick diverse points from front0
            if len(front0) > n_polish:
                cd = self._crowding_distance(F, front0)
                # Select points with highest crowding distance (most diverse)
                top_idx = np.argsort(-cd)[:n_polish]
                polish_idx = [front0[i] for i in top_idx]
            else:
                polish_idx = list(front0)

            for idx in polish_idx:
                if used >= budget or time.time() > deadline:
                    break
                remaining = budget - used
                if remaining < 5:
                    break
                max_evals = min(10, remaining)
                new_x, new_f, evals_used = self._local_polish(
                    problem, bounds, X[idx], lo, hi, rng, max_evals
                )
                used += evals_used - 1  # -1 because problem was already called once in _local_polish
                if np.all(new_f <= F[idx]) and np.any(new_f < F[idx]):
                    X[idx] = new_x
                    F[idx] = new_f

        return X
# EVOLVE-BLOCK-END
