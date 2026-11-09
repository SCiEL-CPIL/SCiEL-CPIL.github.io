"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # NSGA-II Parameters
        # Budget is 200. We need to fit population + generations within this.
        # Population size N = 40. 
        # Initial eval: 40.
        # Remaining: 160.
        # Each generation: N offspring (40 evals).
        # Generations: 4 (40 * 4 = 160). Total 200.
        pop_size = 40
        n_generations = (self.budget - pop_size) // pop_size
        
        # Initialize population randomly (uniform)
        # Using np.random ensures reproducibility with the seed set by evaluator
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of points dominating i
            front = [np.arange(n)]
            
            # Calculate domination relationships
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            # Assign to fronts
            current_front = np.where(n_dom == 0)[0]
            fronts = []
            
            while len(current_front) > 0:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = np.array(next_front)
            
            return fronts

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            m = len(front_indices)
            cd = np.zeros(m)
            F_front = F[front_indices]
            
            for k in range(n_obj):
                # Sort by k-th objective
                sorted_idx = np.argsort(F_front[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                range_k = F_front[sorted_idx[-1], k] - F_front[sorted_idx[0], k]
                if range_k == 0:
                    continue
                
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (F_front[sorted_idx[i+1], k] - F_front[sorted_idx[i-1], k]) / range_k
            
            return cd

        def sbx_crossover(x1, x2, lo, hi):
            eta = 20.0
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
                    else:
                        x[i] = x1[i]
                else:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                    else:
                        x[i] = x1[i]
                # Clip to bounds
                x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        def polynomial_mutation(x, lo, hi):
            eta = 20.0
            for i in range(dim):
                if np.random.rand() < 0.8: # 80% mutation rate per variable
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    
                    range_i = hi[i] - lo[i]
                    x[i] = x[i] + delta * range_i
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Evolution Loop
        for gen in range(n_generations):
            if time.time() > deadline:
                break
            
            # 1. Create offspring
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                # Tournament Selection
                idx1 = np.random.randint(0, pop_size)
                idx2 = np.random.randint(0, pop_size)
                
                # We need ranks for selection. 
                # To avoid recalculating ranks every selection, we can pre-compute them for the current population.
                # However, for simplicity and given small N=40, we can use a simplified selection or pre-compute.
                # Let's pre-compute ranks and crowding for the current population before the loop.
                # Wait, the loop above is creating offspring. I need ranks of X before creating offspring.
                # Let's restructure: Calculate ranks of current X, then select parents.
                
                pass # Placeholder, logic below
                
            # Re-calculate ranks for current population X
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(pop_size, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distance for each front
            cd_all = np.zeros(pop_size)
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd_all[idx] = cd_front[j]
            
            # Now select parents
            for i in range(pop_size):
                # Tournament selection based on rank, then crowding distance
                def tournament():
                    idx = np.random.randint(0, pop_size)
                    return idx
                
                def compare(i1, i2):
                    if ranks[i1] != ranks[i2]:
                        return i1 if ranks[i1] < ranks[i2] else i2
                    else:
                        return i1 if cd_all[i1] > cd_all[i2] else i2
                
                # Two parents
                p1 = tournament()
                p2 = tournament()
                # Simple binary tournament
                if ranks[p1] < ranks[p2]:
                    parent1 = p1
                elif ranks[p2] < ranks[p1]:
                    parent1 = p2
                else:
                    parent1 = p1 if cd_all[p1] > cd_all[p2] else p2
                    
                p3 = tournament()
                p4 = tournament()
                if ranks[p3] < ranks[p4]:
                    parent2 = p3
                elif ranks[p4] < ranks[p3]:
                    parent2 = p4
                else:
                    parent2 = p3 if cd_all[p3] > cd_all[p4] else p4

                # Crossover
                child_x = sbx_crossover(X[parent1], X[parent2], lo, hi)
                # Mutation
                child_x = polynomial_mutation(child_x, lo, hi)
                
                offspring_X[i] = child_x
                offspring_F[i] = problem(child_x)

            # 2. Combine Parent and Offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # 3. Select next generation
            fronts_comb = fast_non_dominated_sort(F_combined)
            
            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            count = 0
            
            for front in fronts_comb:
                if count + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        next_X[count] = X_combined[idx]
                        next_F[count] = F_combined[idx]
                        count += 1
                else:
                    # Need to select some from this front based on crowding distance
                    cd_front = crowding_distance(F_combined, front)
                    # Sort front indices by cd (descending)
                    sorted_indices = np.argsort(-cd_front)
                    needed = pop_size - count
                    for j in range(needed):
                        orig_idx = front[sorted_indices[j]]
                        next_X[count] = X_combined[orig_idx]
                        next_F[count] = F_combined[orig_idx]
                        count += 1
                    break
            
            X = next_X
            F = next_F

        return None
# EVOLVE-BLOCK-END
