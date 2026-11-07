"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(50, max(20, budget // 5))
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, front):
            if len(front) <= 2:
                return np.full(len(front), np.inf)
            cd = np.zeros(len(front))
            for m in range(n_obj):
                f_m = F[front, m]
                sorted_idx = np.argsort(f_m)
                cd[front[sorted_idx[0]]] = np.inf
                cd[front[sorted_idx[-1]]] = np.inf
                f_range = f_m[sorted_idx[-1]] - f_m[sorted_idx[0]]
                if f_range > 0:
                    for i in range(1, len(front) - 1):
                        cd[front[sorted_idx[i]]] += (f_m[sorted_idx[i + 1]] - f_m[sorted_idx[i - 1]]) / f_range
            return cd

        # Main evolutionary loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)

            # Binary tournament selection
            def select_one(F, fronts, cd):
                i1 = np.random.randint(len(F))
                i2 = np.random.randint(len(F))
                # Compare by front rank, then crowding distance
                rank1 = next(idx for idx, front in enumerate(fronts) if i1 in front)
                rank2 = next(idx for idx, front in enumerate(fronts) if i2 in front)
                if rank1 != rank2:
                    return i1 if rank1 < rank2 else i2
                return i1 if cd[i1] > cd[i2] else i2

            # Generate offspring
            for i in range(pop_size):
                p1 = select_one(F, fronts, cd)
                p2 = select_one(F, fronts, cd)
                # Simulated binary crossover
                if np.random.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if np.random.random() < 0.5:
                            child[d] = X[p1, d]
                        else:
                            child[d] = X[p2, d]
                else:
                    child = X[p1].copy()
                # Mutation
                for d in range(dim):
                    if np.random.random() < 1.0 / dim:
                        sigma = 0.2 * (hi[d] - lo[d])
                        child[d] += sigma * np.random.standard_normal()
                child = np.clip(child, lo[d], hi[d])
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                if used >= budget or time.time() > deadline:
                    break

            if used >= budget or time.time() > deadline:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X[:used - pop_size]])
            combined_F = np.vstack([F, offspring_F[:used - pop_size]])
            n_combined = len(combined_F)

            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)

            # Select next generation
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:pop_size - count]:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

            # Recompute fronts and cd for next iteration
            fronts = non_dominated_sort(F)
            # Compute cd for the first front (used in selection)
            if fronts[0]:
                cd = crowding_distance(F, fronts[0])
            else:
                cd = np.zeros(len(F))

        return X
# EVOLVE-BLOCK-END
