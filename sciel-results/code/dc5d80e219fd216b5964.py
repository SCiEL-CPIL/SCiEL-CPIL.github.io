"""Rewritten program: efficient multi-objective optimizer for ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Portfolio-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Track all evaluated points and their objectives
        X = []
        F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            X.append(x.copy())
            F.append(np.array(f))
            used += 1
            return f

        # Phase 1: Random initialization
        n_init = min(20, budget // 5)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # Phase 2: Differential evolution for each objective separately
        # This helps find good solutions for individual objectives
        for obj_idx in range(self.n_obj):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            def single_obj(x):
                f = problem(x)
                return f[obj_idx]

            # Run differential evolution with limited budget
            max_iter = 30
            popsize = 8
            try:
                result = differential_evolution(
                    single_obj,
                    list(zip(lo, hi)),
                    maxiter=max_iter,
                    popsize=popsize,
                    seed=int(rng.integers(0, 10000)),
                    tol=0.01,
                    mutation=(0.5, 1.5),
                    recombination=0.8
                )
                if result.success and result.fun < np.inf:
                    evaluate(result.x)
            except Exception:
                pass

        # Phase 3: NSGA-II style evolutionary search
        pop_size = min(30, max(10, budget // 10))
        
        # Build initial population from existing points
        if len(X) >= pop_size:
            # Select non-dominated subset or random if all dominated
            indices = self._get_non_dominated_indices(F)
            if len(indices) >= pop_size:
                pop_idx = indices[:pop_size]
            else:
                pop_idx = indices + list(rng.choice(len(X), pop_size - len(indices), replace=False))
        else:
            while len(X) < pop_size:
                x = lo + (hi - lo) * rng.random(dim)
                evaluate(x)
            pop_idx = list(range(len(X)))

        X = [X[i] for i in pop_idx]
        F = [F[i] for i in pop_idx]

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)
            
            # Select parents from best fronts
            parents = []
            for front in fronts:
                if len(parents) >= pop_size:
                    break
                parents.extend(front)
            if len(parents) < pop_size:
                parents.extend(fronts[-1][:pop_size - len(parents)])
            parents = parents[:pop_size]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            
            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break
                # Select 3 parents
                p1, p2, p3 = parents[rng.integers(len(parents))], \
                               parents[rng.integers(len(parents))], \
                               parents[rng.integers(len(parents))]
                # Crossover
                child = p1.copy()
                mask = rng.random(dim) < 0.5
                child[mask] = p2[mask]
                # Mutation
                child += 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                f = problem(child)
                offspring_X.append(child)
                offspring_F.append(np.array(f))
                used += 1

            # Combine parent and offspring
            combined_X = X + offspring_X
            combined_F = F + offspring_F
            
            # Selection: keep best pop_size individuals
            fronts = self._fast_non_dominated_sort(combined_F)
            new_idx = []
            for front in fronts:
                if len(new_idx) >= pop_size:
                    break
                if len(new_idx) + len(front) <= pop_size:
                    new_idx.extend(front)
                else:
                    # Need crowding distance to select from this front
                    crowd = self._crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: crowd[i], reverse=True)
                    new_idx.extend(sorted_front[:pop_size - len(new_idx)])
            
            X = [combined_X[i] for i in new_idx]
            F = [combined_F[i] for i in new_idx]

        return X

    def _get_non_dominated_indices(self, F):
        """Get indices of non-dominated points."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return [i for i in range(n) if not dominated[i]]

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
        
        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        if len(indices) <= 2:
            return {i: np.inf for i in indices}
        
        m = len(F[0])
        dist = {i: 0.0 for i in indices}
        
        for k in range(m):
            sorted_idx = sorted(indices, key=lambda i: F[i][k])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0]][k]
            f_max = F[sorted_idx[-1]][k]
            if f_max - f_min < 1e-10:
                continue
            for i in range(1, len(sorted_idx) - 1):
                dist[sorted_idx[i]] += (F[sorted_idx[i + 1]][k] - F[sorted_idx[i - 1]][k]) / (f_max - f_min)
        
        return dist
# EVOLVE-BLOCK-END
