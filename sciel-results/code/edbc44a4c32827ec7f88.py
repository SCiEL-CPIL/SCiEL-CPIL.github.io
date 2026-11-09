"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective NSGA-II optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size: use half the budget for initial population, rest for offspring
        pop_size = min(80, self.budget // 2)
        
        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(fitnesses):
            """Return list of fronts, each front is a list of indices."""
            N = len(fitnesses)
            S = [[] for _ in range(N)]
            n_dom = [0] * N
            rank = [0] * N
            fronts = [[]]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        n_dom[j] += 1

                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(front_indices, fitnesses):
            """Compute crowding distance for a front."""
            N = len(front_indices)
            if N <= 2:
                return [float('inf')] * N

            dist = [0.0] * N
            for m_idx in range(m):
                sorted_idx = sorted(front_indices, key=lambda k: fitnesses[k][m_idx])
                dist[front_indices.index(sorted_idx[0])] = float('inf')
                dist[front_indices.index(sorted_idx[-1])] = float('inf')
                f_min = fitnesses[sorted_idx[0]][m_idx]
                f_max = fitnesses[sorted_idx[-1]][m_idx]
                if f_max == f_min:
                    continue
                for i in range(1, N - 1):
                    idx = front_indices.index(sorted_idx[i])
                    dist[idx] += (fitnesses[sorted_idx[i + 1]][m_idx] - fitnesses[sorted_idx[i - 1]][m_idx]) / (f_max - f_min)

            return dist

        # SBX (Simulated Binary Crossover)
        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated Binary Crossover."""
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(n):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            """Polynomial Mutation."""
            child = x.copy()
            for i in range(n):
                if rng.random() < 1.0 / n:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[i] = x[i] + delta_q * (hi[i] - lo[i])
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Tournament Selection
        def tournament_select(fitnesses, fronts, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            N = len(fitnesses)
            cd = crowding_distance(list(range(N)), fitnesses)
            
            # Create a list of (rank, -crowding_dist) for comparison
            # Lower rank is better, higher crowding distance is better
            indices = list(range(N))
            selected = []
            for _ in range(k):
                idx = rng.integers(N)
                selected.append(idx)
            
            # Compare based on rank first, then crowding distance
            best_idx = selected[0]
            for idx in selected[1:]:
                if fronts.index([best_idx]) < fronts.index([idx]):
                    pass  # best_idx has lower front
                elif fronts.index([best_idx]) > fronts.index([idx]):
                    best_idx = idx
                else:
                    # Same front, compare crowding distance
                    # Need to find crowding distance for each
                    cd_best = cd[fronts[fronts.index([best_idx])].index(best_idx)]
                    cd_idx = cd[fronts[fronts.index([idx])].index(idx)]
                    if cd_idx > cd_best:
                        best_idx = idx
            
            return best_idx

        # Main loop
        while used < self.budget:
            batch = min(pop_size, self.budget - used)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(batch):
                # Select two parents
                p1 = tournament_select(F, fast_non_dominated_sort(F))
                p2 = tournament_select(F, fast_non_dominated_sort(F))
                
                # Crossover
                if rng.random() < 0.9:  # Crossover probability
                    child1, child2 = sbx_crossover(X[p1], X[p2])
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()
                
                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                offspring_X.append(child1)
                offspring_X.append(child2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                used += 2
                
                if used >= self.budget:
                    break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select new population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(front, combined_F)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
