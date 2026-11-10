import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # For small budgets, a smaller population allows more generations.
        # N=20 seems reasonable for 200 evals (10 generations).
        N = 20
        
        # Initialize population
        X = np.random.rand(N, dim) * (hi - lo) + lo
        F = np.array([problem(x) for x in X])
        
        used = N
        
        # Parameters for SBX and PM
        eta_c = 15.0
        eta_m = 20.0
        
        def non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is (N, n_obj)
            """
            N = len(F)
            S = [set() for _ in range(N)]
            n = np.zeros(N, dtype=int)
            rank = np.zeros(N, dtype=int)
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].add(p)
                        n[p] += 1
                        if n[p] == 0:
                            rank[p] = 0
                            
            fronts = []
            i = 0
            current_front = [p for p in range(N) if rank[p] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """
            Calculates crowding distance for individuals in 'indices'.
            Returns dict mapping index -> distance.
            """
            if len(indices) == 0:
                return {}
            if len(indices) == 1:
                return {indices[0]: float('inf')}
            
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                f_vals = np.array([F[i, m] for i in indices])
                min_f = np.min(f_vals)
                max_f = np.max(f_vals)
                if max_f == min_f:
                    continue
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                for k in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[k]
                    idx_prev = sorted_indices[k-1]
                    idx_next = sorted_indices[k+1]
                    dist[idx] += (F[idx_next, m] - F[idx_prev, m]) / (max_f - min_f)
            return dist

        def binary_tournament(F, rank, cd):
            """
            Selects 2 parents using binary tournament.
            rank: array of ranks
            cd: array of crowding distances
            Returns indices of 2 parents.
            """
            i = np.random.randint(N)
            j = np.random.randint(N)
            
            # Compare rank first
            if rank[i] < rank[j]:
                p1 = i
            elif rank[j] < rank[i]:
                p1 = j
            else:
                # Compare crowding distance
                if cd[i] > cd[j]:
                    p1 = i
                else:
                    p1 = j
            
            # Second tournament for p2
            k = np.random.randint(N)
            l = np.random.randint(N)
            if rank[k] < rank[l]:
                p2 = k
            elif rank[l] < rank[k]:
                p2 = l
            else:
                if cd[k] > cd[l]:
                    p2 = k
                else:
                    p2 = l
            
            return p1, p2

        def sbx_crossover(x1, x2, lo, hi):
            """
            Simulated Binary Crossover.
            """
            x1 = np.array(x1)
            x2 = np.array(x2)
            x3 = np.zeros_like(x1)
            x4 = np.zeros_like(x1)
            
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        beta = 1.0
                        if np.random.rand() < 0.5:
                            if x1[i] > x2[i]:
                                beta = (x1[i] - x2[i]) / (hi[i] - lo[i])
                            else:
                                beta = (x2[i] - x1[i]) / (hi[i] - lo[i])
                            if beta > 0:
                                beta_q = ((beta + 1) ** (eta_c + 1)) ** 0.5
                                beta = (beta_q - 1) / beta_q
                        else:
                            if x1[i] < x2[i]:
                                beta = (x1[i] - x2[i]) / (lo[i] - hi[i])
                            else:
                                beta = (x2[i] - x1[i]) / (lo[i] - hi[i])
                            if beta > 0:
                                beta_q = ((beta + 1) ** (eta_c + 1)) ** 0.5
                                beta = (beta_q - 1) / beta_q
                        x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        x3[i] = x1[i]
                        x4[i] = x2[i]
                else:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
                    
            return np.clip(x3, lo, hi), np.clip(x4, lo, hi)

        def polynomial_mutation(x, lo, hi):
            """
            Polynomial Mutation.
            """
            x = np.array(x)
            for i in range(len(x)):
                if np.random.rand() < 0.5:
                    continue
                y = x[i]
                if y <= lo[i] or y >= hi[i]:
                    continue
                delta_1 = (y - lo[i]) / (hi[i] - lo[i])
                delta_2 = (hi[i] - y) / (hi[i] - lo[i])
                rand = np.random.rand()
                if rand < 0.5:
                    xy = 2.0 * rand
                    delta_q = (1.0 + (eta_m + 1.0) * xy) ** (-1.0 / (eta_m + 1.0))
                    delta = delta_q - 1.0
                    y = y + delta * (hi[i] - lo[i]) * delta_1
                else:
                    xy = 1.0 - 2.0 * (rand - 0.5)
                    delta_q = (1.0 + (eta_m + 1.0) * (1.0 - xy)) ** (-1.0 / (eta_m + 1.0))
                    delta = 1.0 - delta_q
                    y = y + delta * (hi[i] - lo[i]) * delta_2
                x[i] = np.clip(y, lo[i], hi[i])
            return x

        # Main Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = np.zeros((N, dim))
            offspring_F = np.zeros((N, n_obj))
            
            for i in range(N):
                p1, p2 = binary_tournament(F, rank, cd)
                # Ensure p1 and p2 are distinct if possible, though not strictly required for SBX
                # But for diversity, let's just proceed.
                
                x1, x2 = X[p1], X[p2]
                c1, c2 = sbx_crossover(x1, x2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
                offspring_F[2*i] = problem(c1)
                offspring_F[2*i+1] = problem(c2)
                
                used += 2
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
                
            # Combine parents and offspring
            X_comb = np.vstack((X, offspring_X))
            F_comb = np.vstack((F, offspring_F))
            
            # Non-dominated sort
            fronts = non_dominated_sort(F_comb)
            
            # Select next generation
            next_X = []
            next_F = []
            rank = np.zeros(len(X_comb), dtype=int)
            cd = np.zeros(len(X_comb))
            
            for i, front in enumerate(fronts):
                if len(next_X) + len(front) <= N:
                    # Add entire front
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                        rank[idx] = i
                else:
                    # Need to select from this front based on crowding distance
                    # Calculate crowding distance for this front
                    front_indices = front
                    cd_dict = crowding_distance(F_comb, front_indices)
                    for idx in front_indices:
                        cd[idx] = cd_dict[idx]
                    
                    # Sort by crowding distance descending
                    sorted_front = sorted(front_indices, key=lambda idx: cd[idx], reverse=True)
                    needed = N - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                        rank[idx] = i
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Update rank and cd for the new population for next iteration's selection
            # We need to re-calculate rank and cd for the new population X, F
            # Actually, the rank and cd arrays above are for the combined population.
            # We need rank and cd for the selected population to use in binary_tournament.
            
            # Re-calculate non-dominated sort for the new population to get ranks
            # And crowding distance for the first front (or all fronts?)
            # Standard NSGA-II uses rank and crowding distance of the current population.
            
            fronts_new = non_dominated_sort(F)
            rank = np.zeros(N, dtype=int)
            cd = np.zeros(N)
            
            for i, front in enumerate(fronts_new):
                for idx in front:
                    rank[idx] = i
                if i == 0:
                    # Calculate crowding distance for the first front (most important for diversity)
                    # Actually, we should calculate for all fronts or at least the ones used in selection.
                    # Let's calculate for all individuals in the population.
                    pass
            
            # Efficiently calculate crowding distance for the whole population?
            # No, crowding distance is usually calculated per front.
            # For tournament selection, we compare rank first. If ranks are equal, we compare CD.
            # So we need CD for individuals in the same front.
            
            for i, front in enumerate(fronts_new):
                if len(front) > 1:
                    cd_dict = crowding_distance(F, front)
                    for idx in front:
                        cd[idx] = cd_dict[idx]
                else:
                    if len(front) == 1:
                        cd[front[0]] = float('inf')

        return X
