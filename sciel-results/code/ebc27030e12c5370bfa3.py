"""NSGA-II based multi-objective optimizer for WFG7 with Nelder-Mead refinement."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II multi-objective optimizer with Nelder-Mead polishing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if _dominates(F[i], F[j]):
                        S[i].append(j)
                    elif _dominates(F[j], F[i]):
                        nd[i] += 1

            current_front = [i for i in range(n) if nd[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def _crowding_distance(F, front_indices):
            """Compute crowding distance for points in a front."""
            m = len(front_indices)
            if m <= 2:
                return [float('inf')] * m

            dist = [0.0] * m
            for k in range(n_obj):
                vals = [F[i][k] for i in front_indices]
                sorted_idx = np.argsort(vals)
                min_val = vals[sorted_idx[0]]
                max_val = vals[sorted_idx[-1]]
                if max_val - min_val < 1e-12:
                    continue
                dist[0] = float('inf')
                dist[m - 1] = float('inf')
                for i in range(1, m - 1):
                    dist[i] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def _evaluate_population(X):
            """Evaluate all points in population."""
            F = np.array([problem(x) for x in X], dtype=np.float64)
            return F

        # NSGA-II parameters
        pop_size = 40
        n_generations = max(1, (budget - pop_size) // (2 * pop_size))

        # Initial population: mix of random and structured points
        X = np.zeros((pop_size, dim))
        # Half random
        X[:pop_size // 2] = lo + (hi - lo) * rng.random((pop_size // 2, dim))
        # Half structured: LHS-like spread
        n_struct = pop_size - pop_size // 2
        for i in range(n_struct):
            for d in range(dim):
                X[pop_size // 2 + i, d] = lo[d] + (hi[d] - lo[d]) * ((i * (dim + 1) + d) % n_struct) / n_struct

        used = 0
        F = _evaluate_population(X)
        used += pop_size

        # Main NSGA-II loop
        for gen in range(n_generations):
            if time.time() > deadline - 60:
                break
            if used >= budget:
                break

            # Create offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                if time.time() > deadline - 60:
                    break
                if used >= budget:
                    break

                # Binary tournament selection
                def tournament_select(pop_X, pop_F, k=2):
                    idxs = rng.choice(len(pop_F), size=k, replace=False)
                    best_idx = idxs[0]
                    best_f = pop_F[best_idx]
                    for idx in idxs[1:]:
                        if _dominates(pop_F[idx], best_f):
                            best_f = pop_F[idx]
                            best_idx = idx
                    return pop_X[best_idx].copy()

                parent1 = tournament_select(X, F)
                parent2 = tournament_select(X, F)

                # SBX crossover
                child = np.zeros(dim)
                for d in range(dim):
                    x1, x2 = parent1[d], parent2[d]
                    if abs(x1 - x2) < 1e-12:
                        child[d] = x1
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 21)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        if rng.random() < 0.5:
                            child[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                child = np.clip(child, lo, hi)

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 11) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 11)
                        child[d] += delta * (hi[d] - lo[d])
                child = np.clip(child, lo, hi)

                offspring_X[i] = child

            # Evaluate offspring
            F_off = _evaluate_population(offspring_X)
            used += pop_size

            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_off])

            # Non-dominated sorting
            fronts = _fast_non_dominated_sort(F_comb)

            # Select next generation
            X_next = []
            F_next = []
            for front in fronts:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    # Need crowding distance to select
                    remaining = pop_size - len(X_next)
                    cd = _crowding_distance(F_comb, front)
                    front_with_cd = [(cd[i], front[i]) for i in range(len(front))]
                    front_with_cd.sort(key=lambda x: -x[0])
                    for _, idx in front_with_cd[:remaining]:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break

            X = np.array(X_next)
            F = np.array(F_next)

        # Extract non-dominated set
        fronts = _fast_non_dominated_sort(F)
        nd_indices = fronts[0]
        archive_X = X[nd_indices]
        archive_F = F[nd_indices]

        # Nelder-Mead refinement on non-dominated points
        if time.time() < deadline - 20 and len(archive_X) > 0:
            n_polish = min(8, len(archive_X), (budget - used) // 50)
            # Select diverse points
            if len(archive_X) > n_polish:
                indices = np.linspace(0, len(archive_X) - 1, n_polish, dtype=int)
            else:
                indices = range(len(archive_X))

            for idx in indices:
                if used >= budget or time.time() > deadline - 10:
                    break

                x0 = archive_X[idx].copy()
                f0 = archive_F[idx].copy()

                # Use Nelder-Mead to minimize sum of objectives
                def objective_sum(x):
                    f = problem(x)
                    used_evals[0] += 1
                    return np.sum(f)

                used_evals = [0]
                try:
                    res = minimize(objective_sum, x0, method='Nelder-Mead',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6,
                                            'adaptive': True})
                    if used_evals[0] > 0:
                        used += used_evals[0]
                        if res.success and used < budget:
                            f_new = problem(res.x)
                            used += 1
                            # Check if non-dominated
                            dominated = False
                            for af in archive_F:
                                if _dominates(af, f_new):
                                    dominated = True
                                    break
                            if not dominated:
                                # Remove dominated points
                                new_archive_F = []
                                new_archive_X = []
                                for i, af in enumerate(archive_F):
                                    if not _dominates(f_new, af):
                                        new_archive_F.append(af)
                                        new_archive_X.append(archive_X[i])
                                new_archive_F.append(f_new)
                                new_archive_X.append(res.x)
                                archive_F = new_archive_F
                                archive_X = new_archive_X
                except Exception:
                    pass

        return list(archive_X)
