"""Multi-objective optimizer for RE34 using NSGA-II with global archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Population size
        pop_size = 40
        n_gen = (budget - pop_size) // pop_size
        if n_gen < 1:
            n_gen = 1
            pop_size = budget // 2
            if pop_size < 2:
                pop_size = 2

        # Initialization: Latin Hypercube Sampling
        X = self._lhs_sample(pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Global Archive
        archive_X = []
        archive_F = []
        
        def update_archive(x_new, f_new):
            # Check if dominated by any in archive
            for af in archive_F:
                if np.all(af <= f_new) and np.any(af < f_new):
                    return
            # Remove dominated
            keep = []
            for i in range(len(archive_X)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                    keep.append(i)
            for i in range(len(archive_X)):
                if i not in keep:
                    archive_X.pop(i)
                    archive_F.pop(i)
            archive_X.append(x_new.copy())
            archive_F.append(f_new.copy())

        for i in range(pop_size):
            update_archive(X[i], F[i])

        # NSGA-II Parameters
        n_crossover = 0.9
        n_mutate = 1.0 / dim
        eta_c = 20.0
        eta_m = 20.0

        def sbx_crossover(x1, x2):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if np.random.rand() < n_crossover:
                    if np.random.rand() < 0.5:
                        beta = 1.0 / (1.0 + 2.0 * (x2[i] - x1[i]) / (hi[i] - lo[i]))
                    else:
                        beta = 1.0 / (1.0 + 2.0 * (x1[i] - x2[i]) / (hi[i] - lo[i]))
                    beta = beta ** (1.0 / (eta_c + 1.0))
                    child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def poly_mutation(x):
            y = x.copy()
            for i in range(dim):
                if np.random.rand() < n_mutate:
                    delta = np.random.rand()
                    if delta < 0.5:
                        val = (1.0 + 2.0 * (delta - 0.5)) ** (1.0 / (eta_m + 1.0))
                    else:
                        val = (1.0 + 2.0 * (0.5 - delta)) ** (1.0 / (eta_m + 1.0))
                    y[i] += (hi[i] - lo[i]) * val
            return np.clip(y, lo, hi)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
                if n_dominates[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            fronts.pop()
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(n_obj):
                obj_vals = F[indices, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val == min_val:
                    continue
                order = np.argsort(obj_vals)
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                for k in range(1, n - 1):
                    dist[order[k]] += (obj_vals[order[k + 1]] - obj_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def lhs_sample(n, dim, lo, hi):
            samples = np.zeros((n, dim))
            for i in range(dim):
                samples[:, i] = lo[i] + (hi[i] - lo[i]) * np.random.permutation(np.linspace(0, 1, n, endpoint=False) + np.random.uniform(0, 1/n, n))
            return samples

        # Main Loop
        while used < budget and time.time() < deadline:
            # Generate Offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            while len(offspring_X) < n_offspring and used + 2 <= budget and time.time() < deadline:
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if i1 == i2:
                    continue
                c1, c2 = sbx_crossover(X[i1], X[i2])
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                update_archive(c1, f1)
                update_archive(c2, f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:n_offspring])
            offspring_F = np.array(offspring_F[:n_offspring])

            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Sort
            fronts = fast_non_dominated_sort(combined_F)
            
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate
                    needed = pop_size - len(next_X)
                    dists = crowding_distance(combined_F, front)
                    order = np.argsort(dists)[::-1]
                    for i in range(needed):
                        idx = front[order[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _lhs_sample(self, n, dim, lo, hi):
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = lo[i] + (hi[i] - lo[i]) * (np.random.permutation(np.linspace(0, 1, n, endpoint=False) + np.random.uniform(0, 1/n, n)))
        return samples
# EVOLVE-BLOCK-END
