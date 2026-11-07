# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj

        # Use the global RNG (seeded by evaluator)
        rng = np.random

        # Population size
        N = 20
        # Crossover/mutation parameters
        eta_c = 20.0
        eta_m = 20.0
        prob_cx = 0.9

        def fast_non_dominated_sort(F):
            """Returns list of fronts (each a list of indices) and rank array."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1

            rank = np.zeros(n, dtype=int)
            current_front = [i for i in range(n) if nd[i] == 0]
            front_idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = front_idx + 1
                            next_front.append(j)
                current_front = next_front
                front_idx += 1

            # Assign ranks to all
            for i in range(n):
                if rank[i] == 0 and i not in [f for front in fronts for f in front]:
                    rank[i] = front_idx - 1

            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            m = n_obj
            n = len(indices)
            cd = np.zeros(n)

            for k in range(m):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range

            return cd

        def SBX_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if rng.random() > prob_cx:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()

            for i in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = rng.random()
                        beta = 1.0 / (1.0 + 2.0 * (1.0 - u)) if u <= 0.5 else 1.0 / (1.0 + 2.0 * u)
                        beta = beta ** (1.0 / (eta_c + 1.0))
                        c1 = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        c2 = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2

            return child1, child2

        def polynomial_mutation(x):
            """Polynomial mutation."""
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N

        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sorting
            fronts, rank = fast_non_dominated_sort(F)

            # Create mating pool by truncating fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) <= N:
                    mating_pool.extend(front)
                else:
                    # Need to select some from this front based on crowding distance
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    mating_pool.extend(sorted_front[:N - len(mating_pool)])
                    break

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < N:
                if time.time() > deadline:
                    break

                # Select parents using tournament selection on rank and crowding distance
                p1 = self._tournament_select(mating_pool, F, rank)
                p2 = self._tournament_select(mating_pool, F, rank)

                c1, c2 = SBX_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                f1 = problem(c1)
                f2 = problem(c2)
                used += 2

                offspring_X.append(c1)
                offspring_F.append(f1)
                if len(offspring_X) < N:
                    offspring_X.append(c2)
                    offspring_F.append(f2)

            if not offspring_X:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            total = len(combined_X)

            # Fast non-dominated sorting on combined
            fronts, rank = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            new_indices = []

            for front in fronts:
                if len(new_indices) + len(front) <= N:
                    new_indices.extend(front)
                else:
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    new_indices.extend(sorted_front[:N - len(new_indices)])
                    break

            X = combined_X[new_indices]
            F = combined_F[new_indices]

        return X

    def _tournament_select(self, pool, F, rank):
        """Binary tournament selection."""
        i = np.random.randint(len(pool))
        j = np.random.randint(len(pool))
        ri = rank[pool[i]]
        rj = rank[pool[j]]
        if ri < rj:
            return pool[i]
        elif rj < ri:
            return pool[j]
        else:
            # Same rank, use crowding distance
            # Simplified: random
            return pool[i] if np.random.random() < 0.5 else pool[j]
# EVOLVE-BLOCK-END
