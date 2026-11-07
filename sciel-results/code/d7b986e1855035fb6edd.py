"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: balance diversity and convergence
        pop_size = min(50, max(20, budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        def fast_non_dominated_sort(F):
            """Assign non-dominated ranks to a set of objective vectors."""
            n_pts = len(F)
            S = [[] for _ in range(n_pts)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n_pts, dtype=int)  # number of points that dominate i
            rank = np.zeros(n_pts, dtype=int)

            for i in range(n_pts):
                for j in range(n_pts):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n_pts) if n_dom[i] == 0]
            if not front:
                front = [0]  # fallback
            rank_idx = 0
            while front:
                next_front = []
                for i in front:
                    rank[i] = rank_idx
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front
                rank_idx += 1

            # Assign max rank to any unassigned
            max_rank = rank_idx - 1
            for i in range(n_pts):
                if rank[i] == 0 and i not in [0]:
                    rank[i] = max_rank
            return rank

        def crowding_distance(F, rank):
            """Compute crowding distance for each point based on its rank."""
            n_pts = len(F)
            cd = np.zeros(n_pts)
            if n_pts <= 2:
                cd[:] = np.inf
                return cd

            for r in range(int(rank.max()) + 1):
                members = np.where(rank == r)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue

                for m in range(n_obj):
                    sorted_idx = members[np.argsort(F[members, m])]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_min = F[sorted_idx[0], m]
                    f_max = F[sorted_idx[-1], m]
                    if f_max - f_min < 1e-15:
                        continue
                    for k in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[k]] += (
                            (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / (f_max - f_min)
                        )
            return cd

        def binary_tournament(rank, cd):
            """Select a parent using binary tournament on rank and crowding distance."""
            i, j = rng.integers(len(rank)), rng.integers(len(rank))
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if cd[i] > cd[j] else j

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20
            child1 = np.zeros(n)
            child2 = np.zeros(n)
            for k in range(n):
                if rng.random() < 0.5:
                    beta = 1.0 / (1.0 + 2.0 * rng.random())
                    alpha1 = 2.0 * beta
                    alpha2 = 2.0 * (1.0 - beta)
                    if alpha1 <= 1.0:
                        beta1_q = (0.5 * (alpha1 + 1.0)) ** (1.0 / (eta + 1.0))
                    else:
                        beta1_q = (0.5 / (2.0 - alpha1)) ** (1.0 / (eta + 1.0))
                    if alpha2 <= 1.0:
                        beta2_q = (0.5 * (alpha2 + 1.0)) ** (1.0 / (eta + 1.0))
                    else:
                        beta2_q = (0.5 / (2.0 - alpha2)) ** (1.0 / (eta + 1.0))
                    child1[k] = 0.5 * ((1.0 + beta1_q) * x1[k] + (1.0 - beta1_q) * x2[k])
                    child2[k] = 0.5 * ((1.0 + beta2_q) * x2[k] + (1.0 - beta2_q) * x1[k])
                else:
                    child1[k] = x2[k]
                    child2[k] = x1[k]
            return child1, child2

        def mutate(x, sigma=0.1):
            """Polynomial mutation."""
            eta = 20
            x_mut = x.copy()
            for k in range(n):
                if rng.random() < 0.5:
                    lo_k, hi_k = lo[k], hi[k]
                    ux = (x[k] - lo_k) / (hi_k - lo_k + 1e-15)
                    beta = 1.0 + 2.0 * min(ux, 1.0 - ux)
                    alpha = 2.0 + beta * (beta - 2.0)
                    if rng.random() < 0.5:
                        delta1 = (2.0 * rng.random() + (1.0 - 1.0 / (alpha + 1.0))) ** (1.0 / (eta + 1.0)) - 1.0
                        delta = -delta1
                    else:
                        delta1 = (2.0 * (1.0 - rng.random()) + (1.0 - 1.0 / (alpha + 1.0))) ** (1.0 / (eta + 1.0)) - 1.0
                        delta = delta1
                    x_mut[k] = x[k] + delta * (hi_k - lo_k)
                    x_mut[k] = np.clip(x_mut[k], lo_k, hi_k)
            return x_mut

        # Main evolutionary loop
        while evals < budget:
            if time.time() > deadline:
                break

            # Evaluate current population
            rank = fast_non_dominated_sort(F)
            cd = crowding_distance(F, rank)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            for _ in range(n_offspring // 2):
                if time.time() > deadline:
                    break
                p1 = binary_tournament(rank, cd)
                p2 = binary_tournament(rank, cd)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.extend([c1, c2])
                offspring_F.extend([problem(c1), problem(c2)])
                evals += 2

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            combined_rank = fast_non_dominated_sort(combined_F)
            combined_cd = crowding_distance(combined_F, combined_rank)

            # Sort by rank, then by crowding distance
            order = np.lexsort((combined_cd * -1, combined_rank))
            new_idx = order[:pop_size]

            X = combined_X[new_idx]
            F = combined_F[new_idx]

        return X
# EVOLVE-BLOCK-END
