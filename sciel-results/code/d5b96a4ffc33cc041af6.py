"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size
        # We have 200 budget. 
        # Init: 20.
        # Generations: (200-20)/20 = 9 generations of 20 offspring.
        pop_size = 20
        n_generations = (self.budget - pop_size) // pop_size
        
        # 1. Initialization
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            # Compute domination
            # F[i] dominates F[j] if all F[i] <= F[j] and any F[i] < F[j]
            # Vectorized check:
            # D[i, j] is True if i dominates j
            # F[i] <= F[j] for all obj
            # np.all(F[i][:, None] <= F[j][None, :], axis=0)
            
            # This is O(N^2 * M). N=40, M=3. Very fast.
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[j] += 1 # Wait, standard algo counts dominated points for i, and dominators for j?
                        # Standard:
                        # S[i] = set of points dominated by i
                        # n[i] = number of points dominating i
                        
            # Correct Standard Algorithm:
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int) # Number of points dominating i
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
                
            return rank, S

        # Helper: Crowding Distance
        def crowding_distance(F, rank, S):
            n = len(F)
            cd = np.zeros(n)
            # Group by rank
            ranks = np.unique(rank)
            for r in ranks:
                idx = np.where(rank == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    # Sort by objective m
                    sorted_idx = idx[np.argsort(F[idx, m])]
                    min_val = F[sorted_idx[0], m]
                    max_val = F[sorted_idx[-1], m]
                    if max_val - min_val < 1e-10:
                        cd[sorted_idx] = np.inf
                        continue
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (max_val - min_val)
            return cd

        # 2. Evolution Loop
        for g in range(n_generations):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
                
            # Generate offspring
            C_X = np.zeros((pop_size, self.dim))
            C_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if used >= self.budget:
                    break
                # Select 3 distinct parents
                p1_idx, p2_idx, p3_idx = rng.choice(pop_size, 3, replace=False)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                p3 = X[p3_idx]
                
                # Differential Evolution Mutation
                # F factor
                F_factor = 0.5 + 0.5 * rng.random()
                child = p1 + F_factor * (p2 - p3)
                
                # Clip
                child = np.clip(child, lo, hi)
                
                # Evaluate
                C_X[i] = child
                C_F[i] = problem(child)
                used += 1
            
            # Combine Parent and Child
            Q_X = np.vstack([X, C_X])
            Q_F = np.vstack([F, C_F])
            
            # Non-dominated sort
            rank, S = fast_non_dominated_sort(Q_F)
            cd = crowding_distance(Q_F, rank, S)
            
            # Select best pop_size individuals
            # Sort by rank (ascending), then crowding distance (descending)
            # We need to pick the first pop_size from the sorted list
            # Create indices sorted by (rank, -cd)
            sort_keys = np.lexsort((-cd, rank))
            selected_indices = sort_keys[:pop_size]
            
            X = Q_X[selected_indices]
            F = Q_F[selected_indices]
            
        return X
# EVOLVE-BLOCK-END
