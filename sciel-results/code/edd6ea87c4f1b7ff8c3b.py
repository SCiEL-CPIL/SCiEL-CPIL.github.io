"""NSGA-II multi-objective optimizer for WFG9 with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with crowding distance, tournament selection, and SBX/PMX."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 285.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # --- NSGA-II parameters ---
        N = 60  # population size
        eta_c = 20.0  # SBX distribution index
        eta_m = 20.0  # polynomial mutation distribution index
        pm = 1.0 / dim  # mutation probability per gene
        tournament_size = 2

        # --- Initialize population with diverse points ---
        pop = lo + span * rng.random((N, dim))
        F = np.empty((N, n_obj))
        evals = 0
        all_F_list = []

        for i in range(N):
            F[i] = problem(pop[i])
            evals += 1
            all_F_list.append(F[i].copy())

        # --- Fast non-dominated sort ---
        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts (list of indices) sorted by dominance rank."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1

            return fronts, rank

        # --- Crowding distance ---
        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a subset of solutions."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf

            F_sub = F_arr[indices]
            n = len(indices)
            cd = np.zeros(n)

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (
                        F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]
                    ) / (f_max - f_min)

            return cd

        # --- SBX crossover ---
        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        continue
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # --- Polynomial mutation ---
        def polynomial_mutation(x):
            """Polynomial mutation."""
            x = x.copy()
            for i in range(dim):
                if rng.random() < pm:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    x[i] += delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # --- Tournament selection ---
        def tournament_select(pop, F, rank, cd, n_select=1):
            """Tournament selection based on rank and crowding distance."""
            selected = []
            for _ in range(n_select):
                idx1 = rng.integers(N)
                idx2 = rng.integers(N)
                if rank[idx1] < rank[idx2]:
                    selected.append(idx1)
                elif rank[idx2] < rank[idx1]:
                    selected.append(idx2)
                else:
                    if cd[idx1] > cd[idx2]:
                        selected.append(idx1)
                    else:
                        selected.append(idx2)
            return selected

        # --- Main NSGA-II loop ---
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            # Create offspring population
            offspring = []
            offspring_F = []
            offspring_count = 0

            while offspring_count < N and evals < budget:
                # Select parents
                p1_idx, p2_idx = tournament_select(pop, F, *fast_non_dominated_sort(F), 2)
                # Wait, that's inefficient. Let's precompute rank and cd.
                break

            # Precompute rank and crowding distance for current population
            fronts, rank = fast_non_dominated_sort(F)
            # Compute crowding distance for each front
            cd = np.zeros(N)
            for front in fronts:
                cd[front] = crowding_distance(F, front)

            # Generate offspring
            offspring = []
            offspring_F = []
            offspring_count = 0

            while offspring_count < N and evals < budget:
                # Select parents using tournament
                idx1 = rng.integers(N)
                idx2 = rng.integers(N)
                if rank[idx1] < rank[idx2]:
                    p1 = idx1
                elif rank[idx2] < rank[idx1]:
                    p1 = idx2
                else:
                    if cd[idx1] > cd[idx2]:
                        p1 = idx1
                    else:
                        p1 = idx2

                idx3 = rng.integers(N)
                idx4 = rng.integers(N)
                if rank[idx3] < rank[idx4]:
                    p2 = idx3
                elif rank[idx4] < rank[idx3]:
                    p2 = idx4
                else:
                    if cd[idx3] > cd[idx4]:
                        p2 = idx3
                    else:
                        p2 = idx4

                # Crossover
                child1, child2 = sbx_crossover(pop[p1], pop[p2])

                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)

                # Evaluate child1
                f1 = problem(child1)
                evals += 1
                all_F_list.append(f1.copy())
                offspring.append(child1)
                offspring_F.append(f1)
                offspring_count += 1

                if evals >= budget:
                    break

                # Evaluate child2
                f2 = problem(child2)
                evals += 1
                all_F_list.append(f2.copy())
                offspring.append(child2)
                offspring_F.append(f2)
                offspring_count += 1

            if offspring_count == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([pop, np.array(offspring)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            N_comb = 2 * N

            # Fast non-dominated sort on combined
            fronts_comb, rank_comb = fast_non_dominated_sort(combined_F)

            # Compute crowding distance for each front
            cd_comb = np.zeros(N_comb)
            for front in fronts_comb:
                cd_comb[front] = crowding_distance(combined_F, front)

            # Select next generation
            new_pop = []
            new_F = []
            new_count = 0
            for front in fronts_comb:
                if new_count + len(front) <= N:
                    for idx in front:
                        new_pop.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                        new_count += 1
                else:
                    # Sort front by crowding distance (descending)
                    front_cd = cd_comb[front]
                    sorted_front = np.argsort(-front_cd)
                    remaining = N - new_count
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_pop.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                        new_count += 1
                    break

            pop = np.array(new_pop)
            F = np.array(new_F)

        return pop

# EVOLVE-BLOCK-END
