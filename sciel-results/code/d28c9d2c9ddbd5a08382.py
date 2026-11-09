import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        start_time = time.time()
        deadline = 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Hyperparameters
        pop_size = 50
        if self.budget < 100:
            pop_size = max(10, self.budget // 4)
            
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability
        p_m = 1.0 / self.dim  # Mutation probability per gene

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where fronts[i] is a list of indices in F
            that are in the i-th non-dominated front.
            """
            n = len(F)
            S = [set() for _ in range(n)]  # Dominates
            n_dominated = [0] * n          # Number of individuals dominating i
            rank = [0] * n                 # Rank (front index)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0
                
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a specific front.
            Returns a dictionary mapping index to distance.
            """
            if len(front_indices) == 0:
                return {}
            if len(front_indices) == 1:
                return {front_indices[0]: float('inf')}
            
            dist = {i: 0.0 for i in front_indices}
            n_obj = F.shape[1]
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_front = sorted(front_indices, key=lambda i: F[i, m])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                if f_max == f_min:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    prev_idx = sorted_front[i - 1]
                    next_idx = sorted_front[i + 1]
                    dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / (f_max - f_min)
            
            return dist

        def sbx_crossover(X1, X2):
            """
            Simulated Binary Crossover.
            X1, X2: arrays of shape (dim,)
            Returns two children.
            """
            C1 = np.zeros_like(X1)
            C2 = np.zeros_like(X2)
            
            if np.random.rand() < p_c:
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        if (X1[i] - X2[i]) == 0:
                            beta = 1.0
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                        
                        C1[i] = 0.5 * ((1.0 + beta) * X1[i] + (1.0 - beta) * X2[i])
                        C2[i] = 0.5 * ((1.0 - beta) * X1[i] + (1.0 + beta) * X2[i])
                    else:
                        C1[i] = X1[i]
                        C2[i] = X2[i]
            else:
                C1 = X1.copy()
                C2 = X2.copy()
                
            return C1, C2

        def polynomial_mutation(x):
            """
            Polynomial mutation.
            """
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    x_min = lo[i]
                    x_max = hi[i]
                    xmut1 = x[i] + delta1 * (x_max - x_min)
                    xmut2 = x[i] + delta2 * (x_max - x_min)
                    if np.random.rand() < 0.5:
                        x_mut[i] = np.clip(xmut1, x_min, x_max)
                    else:
                        x_mut[i] = np.clip(xmut2, x_min, x_max)
            return x_mut

        def select_parent(ranks, crowding, n):
            """Binary tournament selection."""
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            if ranks[idx1] < ranks[idx2]:
                return idx1
            elif ranks[idx2] < ranks[idx1]:
                return idx2
            else:
                if crowding[idx1] > crowding[idx2]:
                    return idx1
                elif crowding[idx2] > crowding[idx1]:
                    return idx2
                else:
                    return idx1 if np.random.rand() < 0.5 else idx2

        # Main Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sort
            fronts, ranks = fast_non_dominated_sort(F)
            
            # 2. Crowding distance
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 1:
                    cdist = crowding_distance(F, front)
                    for idx, val in cdist.items():
                        crowding[idx] = val

            # 3. Generate offspring
            offspring_X = []
            offspring_F = []
            
            # Generate pop_size offspring
            for _ in range(pop_size // 2):
                if used >= self.budget:
                    break
                
                p1_idx = select_parent(ranks, crowding, len(F))
                p2_idx = select_parent(ranks, crowding, len(F))
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= self.budget:
                    break
                
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            # 4. Selection
            # Combine
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Sort combined
            combined_fronts, combined_ranks = fast_non_dominated_sort(combined_F)
            
            # Select next population
            next_X = []
            next_F = []
            
            # Calculate crowding for combined fronts as we fill
            remaining = pop_size
            for front in combined_fronts:
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    # Add all
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining -= len(front)
                else:
                    # Need to select some from this front based on crowding
                    # Calculate crowding for this specific front in the combined population
                    cdist = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_indices = sorted(front, key=lambda i: cdist[i], reverse=True)
                    for idx in sorted_indices[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining = 0

            X = np.array(next_X)
            F = np.array(next_F)

        # Return best set (ignored by evaluator but good practice)
        # We return the last population
        return X

    def _tournament_select(self, F, ranks, crowding, n):
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        if ranks[idx1] < ranks[idx2]:
            return idx1
        elif ranks[idx2] < ranks[idx1]:
            return idx2
        else:
            if crowding[idx1] > crowding[idx2]:
                return idx1
            elif crowding[idx2] > crowding[idx1]:
                return idx2
            else:
                return idx1 if np.random.rand() < 0.5 else idx2
