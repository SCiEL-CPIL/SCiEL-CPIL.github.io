import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 20
        eta_c = 15.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        
        # Helper: Check if a is dominated by b
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            S = [set() for _ in range(len(F))]  # Dominates
            n = [0 for _ in range(len(F))]  # Number of dominators
            ranks = [0] * len(F)
            
            for i in range(len(F)):
                for j in range(i + 1, len(F)):
                    if dominates(F[i], F[j]):
                        S[i].add(j)
                        n[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].add(i)
                        n[i] += 1
            
            current_front = []
            for i in range(len(F)):
                if n[i] == 0:
                    current_front.append(i)
                    ranks[i] = 0
            
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return ranks, S
        
        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            distances = np.zeros(len(F))
            unique_ranks = sorted(set(ranks))
            for r in unique_ranks:
                idxs = [i for i in range(len(F)) if ranks[i] == r]
                if len(idxs) <= 2:
                    for i in idxs:
                        distances[i] = float('inf')
                    continue
                
                for m in range(self.n_obj):
                    obj_vals = F[idxs, m]
                    min_val = np.min(obj_vals)
                    max_val = np.max(obj_vals)
                    if max_val - min_val == 0:
                        continue
                    # Sort indices by objective m
                    sorted_idxs = sorted(idxs, key=lambda i: F[i, m])
                    # End points
                    distances[sorted_idxs[0]] = float('inf')
                    distances[sorted_idxs[-1]] = float('inf')
                    for k in range(1, len(sorted_idxs) - 1):
                        i_prev = sorted_idxs[k-1]
                        i_next = sorted_idxs[k+1]
                        i_curr = sorted_idxs[k]
                        distances[i_curr] += (F[i_next, m] - F[i_prev, m]) / (max_val - min_val)
            return distances
        
        # Helper: Tournament Selection
        def tournament_select(pop, F, ranks, distances, k=2):
            candidates = np.random.randint(0, len(pop), size=k)
            best_idx = candidates[0]
            for i in range(1, k):
                idx = candidates[i]
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx]:
                    if distances[idx] > distances[best_idx]:
                        best_idx = idx
            return pop[best_idx]
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta):
            x3 = np.zeros(self.dim)
            x4 = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if np.random.rand() <= 1.0 / (eta + 1):
                            beta = (2 * np.random.rand()) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1))
                        x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        x3[i] = x1[i]
                        x4[i] = x2[i]
                else:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
            return x3, x4
        
        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta, lo, hi):
            x_new = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:  # Mutation probability 1/dim? Or 0.5? Let's use 1/dim for sparsity or 0.5 for exploration. 1/dim is standard.
                    if np.random.rand() < 1.0 / self.dim:
                        if np.random.rand() <= 2.0 / (eta + 1):
                            delta1 = (2 * np.random.rand()) ** (1.0 / (eta + 1)) - 1
                            beta = 1 + delta1 * (x[i] - lo[i]) / (hi[i] - lo[i])
                        else:
                            delta2 = 1 - (2 * (1 - np.random.rand())) ** (1.0 / (eta + 1))
                            beta = 1 + delta2 * (hi[i] - x[i]) / (hi[i] - lo[i])
                        x_new[i] = x[i] + beta * (hi[i] - lo[i])
            x_new = np.clip(x_new, lo, hi)
            return x_new

        # 1. Initialization
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # 2. Evolution
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # Number of offspring per generation
            # Let's do 10 offspring per generation to keep overhead low
            n_offspring = min(10, self.budget - used)
            
            for _ in range(n_offspring):
                # Select parents
                ranks, S = non_dominated_sort(F)
                distances = crowding_distance(F, ranks)
                
                p1 = tournament_select(X, F, ranks, distances)
                p2 = tournament_select(X, F, ranks, distances)
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2, eta_c)
                
                # Mutation
                c1 = polynomial_mutation(c1, eta_m, lo, hi)
                c2 = polynomial_mutation(c2, eta_m, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                if used >= self.budget:
                    break
                f2 = problem(c2)
                used += 1
                if used >= self.budget:
                    break
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if not offspring_X:
                break
            
            # Combine population and offspring
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort
            ranks, S = non_dominated_sort(F_new)
            distances = crowding_distance(F_new, ranks)
            
            # Select best pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            indices = np.arange(len(X_new))
            # Create a sort key: (rank, -crowding_distance)
            # We want lowest rank, highest crowding
            sort_keys = [(ranks[i], -distances[i]) for i in indices]
            sorted_indices = [indices[i] for i in np.argsort([k[0] for k in sort_keys], kind='mergesort')]
            # Wait, argsort only sorts by first element if we pass a list of tuples? No, np.argsort doesn't handle tuples directly for multi-key in one call easily without structured arrays.
            # Let's do a stable sort by rank, then sort each rank group by crowding.
            
            unique_ranks = sorted(set(ranks))
            selected_indices = []
            for r in unique_ranks:
                if len(selected_indices) >= pop_size:
                    break
                idxs_in_rank = [i for i in range(len(X_new)) if ranks[i] == r]
                # Sort by crowding distance descending
                idxs_in_rank.sort(key=lambda i: distances[i], reverse=True)
                
                needed = pop_size - len(selected_indices)
                selected_indices.extend(idxs_in_rank[:needed])
            
            # Update population
            X = X_new[selected_indices]
            F = F_new[selected_indices]
            
            # If we ran out of budget in the inner loop, we break the outer loop next check
            if used >= self.budget:
                break

        return X
