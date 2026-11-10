"""Multi-objective optimizer on WFG4 with refined NSGA-II + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style optimizer for WFG4 with LHS init and local polish."""

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

    def _local_polish(self, problem, x, F, bounds, budget_left, dim, rng):
        """Local polish: Nelder-Mead style search around a point."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        best_x = x.copy()
        best_F = F.copy()
        step = 0.1 * (hi - lo)
        evals = 0

        while evals < budget_left and step > 1e-6:
            improved = False
            for d in range(dim):
                for direction in [1, -1]:
                    if evals >= budget_left:
                        break
                    trial = best_x.copy()
                    trial[d] += direction * step[d]
                    trial = np.clip(trial, lo, hi)
                    f_trial = problem(trial)
                    evals += 1
                    # Check if this improves (dominates or is non-dominated)
                    if np.all(f_trial <= best_F) and np.any(f_trial < best_F):
                        best_x = trial
                        best_F = f_trial
                        improved = True
            if improved:
                step *= 0.7
            else:
                step *= 0.5
        return best_x, best_F, evals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Population size - smaller to leave budget for polish
        pop_size = min(30, max(15, budget // 6))

        # Initialize population with LHS for better coverage
        X = self._lhs_sample(rng, pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reserve budget for local polish (last 30% of evaluations)
        polish_budget = max(10, budget // 4)
        de_budget = budget - polish_budget

        # DE parameters
        F_CR = 0.8
        CR = 0.9
        K = 0.5

        while used < de_budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Selection: pick parents from best front using tournament
            front0 = fronts[0]
            if len(front0) < 2:
                # Not enough in front0, use whole population
                parent_pool = list(range(pop_size))
            else:
                parent_pool = front0

            # Pick 3 distinct parents for DE mutation
            if len(parent_pool) >= 3:
                p1, p2, p3 = rng.choice(parent_pool, size=3, replace=False)
            else:
                # Fallback: random selection with replacement
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
            # Check if trial dominates any existing point
            dominated_by_trial = []
            for i in range(pop_size):
                if np.all(ft <= F[i]) and np.any(ft < F[i]):
                    dominated_by_trial.append(i)

            if dominated_by_trial:
                # Replace the one with worst crowding distance among dominated
                sums = np.sum(F[dominated_by_trial], axis=1)
                worst_idx = dominated_by_trial[np.argmax(sums)]
                X[worst_idx] = trial
                F[worst_idx] = ft
            else:
                # Check if trial is dominated by existing points
                dominates_trial = []
                for i in range(pop_size):
                    if np.all(F[i] <= ft) and np.any(F[i] < ft):
                        dominates_trial.append(i)

                if dominates_trial:
                    # Trial is dominated; only replace if it improves diversity
                    worst_front = fronts[-1]
                    if worst_front:
                        cd = self._crowding_distance(F, worst_front)
                        worst_pos = np.argmin(cd)
                        worst_idx = worst_front[worst_pos]
                        if np.sum(ft) < np.sum(F[worst_idx]):
                            X[worst_idx] = trial
                            F[worst_idx] = ft
                else:
                    # Trial is non-dominated and doesn't dominate anyone
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

        # Local polish phase: refine the best non-dominated points
        if used < budget and time.time() < deadline:
            # Get the first front (best non-dominated)
            fronts = self._fast_non_dominated_sort(F)
            front0 = fronts[0] if fronts else list(range(pop_size))

            # Polish each point in the first front, distributing remaining budget
            remaining = budget - used
            n_polish = len(front0)
            if n_polish > 0:
                budget_per_point = max(1, remaining // n_polish)
                for idx in front0:
                    if used >= budget or time.time() > deadline:
                        break
                    bx, bf, evals_used = self._local_polish(
                        problem, X[idx], F[idx], bounds,
                        min(budget_per_point, budget - used), dim, rng
                    )
                    used += evals_used
                    # Only replace if improved
                    if np.all(bf <= F[idx]) and np.any(bf < F[idx]):
                        X[idx] = bx
                        F[idx] = bf

        return X
# EVOLVE-BLOCK-END
