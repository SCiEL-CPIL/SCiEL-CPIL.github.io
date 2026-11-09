"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a lightweight NSGA-II approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320  # Safety margin before 330s kill

        rng = np.random
        dim, budget, n_obj = self.dim, self.budget, self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 40
        if budget < pop_size:
            pop_size = budget
            
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
                
            # Remove empty last front if any
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(F[indices[0]])
            cd = [0.0] * len(indices)
            
            for k in range(m):
                min_val = np.min(F[indices, k])
                max_val = np.max(F[indices, k])
                if max_val == min_val:
                    continue
                sorted_idx = np.argsort(F[indices, k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                for i in range(1, len(indices) - 1):
                    cd[sorted_idx[i]] += (F[indices[sorted_idx[i+1]], k] - F[indices[sorted_idx[i-1]], k]) / (max_val - min_val)
                    
            return cd

        def tournament_select(F, fronts, k=2):
            # Assign rank
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
                    
            while True:
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if ranks[a] < ranks[b]:
                    return a
                elif ranks[b] < ranks[a]:
                    return b
                else:
                    # Tie break with crowding distance if needed, or random
                    # For simplicity, just pick random if ranks are equal
                    return a

        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1, x2
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        def polynomial_mutation(x, eta=20.0):
            child = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[i] += delta * (hi[i] - lo[i])
            return np.clip(child, lo, hi)

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = []
            for _ in range(pop_size):
                p1 = tournament_select(F, fast_non_dominated_sort(F))
                p2 = tournament_select(F, fast_non_dominated_sort(F))
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # We need to evaluate them. We have limited budget.
                # We can only evaluate as many as we have budget for.
                if used < budget:
                    f1 = problem(c1)
                    used += 1
                    offspring_X.append((c1, f1))
                if used < budget:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append((c2, f2))
                else:
                    break
            
            if not offspring_X:
                break

            # Combine
            X_comb = np.vstack([X, np.array([o[0] for o in offspring_X])])
            F_comb = np.vstack([F, np.array([o[1] for o in offspring_X])])
            
            # Select next population
            fronts = fast_non_dominated_sort(F_comb)
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to trim this front
                    needed = pop_size - len(next_X)
                    cd = crowding_distance(F_comb, front)
                    sorted_front = np.argsort(cd)[::-1]
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
# EVOLVE-BLOCK-END
