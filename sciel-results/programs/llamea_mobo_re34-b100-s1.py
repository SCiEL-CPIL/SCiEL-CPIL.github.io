"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution (DE).
    Uses DE/rand/1 mutation and non-dominated sorting for selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Set population size to allow exactly full budget usage if possible
        # Budget 200, Dim 5. 
        # Let's try pop_size = 20.
        # Init: 20. Remaining: 180. Generations: 9. Total: 200.
        pop_size = 20
        if self.budget < pop_size:
            pop_size = max(2, self.budget)
            
        # 1. Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Non-dominated Sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            for i in range(n):
                if n_dom[i] == 0:
                    ranks[i] = 0
                    front.append(i)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return ranks

        # Helper: Calculate Crowding Distance
        def crowding_distance(F, ranks):
            n = len(F)
            if n == 0:
                return np.array([])
            distances = np.zeros(n)
            for r in np.unique(ranks):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    distances[indices] = np.inf
                    continue
                for m in range(self.n_obj):
                    obj_vals = F[indices, m]
                    min_val = np.min(obj_vals)
                    max_val = np.max(obj_vals)
                    if max_val == min_val:
                        distances[indices] = np.inf
                        continue
                    sorted_indices = indices[np.argsort(obj_vals)]
                    distances[sorted_indices[0]] = np.inf
                    distances[sorted_indices[-1]] = np.inf
                    for i in range(1, len(sorted_indices) - 1):
                        prev_idx = sorted_indices[i-1]
                        next_idx = sorted_indices[i+1]
                        curr_idx = sorted_indices[i]
                        distances[curr_idx] += (F[next_idx, m] - F[prev_idx, m]) / (max_val - min_val)
            return distances

        # Main DE Loop
        while used + pop_size <= self.budget:
            # Generate Offspring using DE/rand/1
            children_X = np.empty_like(X)
            children_F = np.empty_like(F)
            
            for i in range(pop_size):
                # Select 3 distinct parents
                idxs = rng.choice(pop_size, 3, replace=False)
                a, b, c = X[idxs[0]], X[idxs[1]], X[idxs[2]]
                
                # Mutation
                F_cr = 0.5  # Crossover rate? No, F is scaling factor. Let's use standard DE notation.
                # Standard DE: v = a + F * (b - c)
                F_factor = 0.5
                v = a + F_factor * (b - c)
                
                # Crossover (Binomial Crossover)
                j_rand = rng.integers(0, self.dim)
                u = rng.random(self.dim)
                child = np.where(u <= 0.5, v, X[i])
                child[j_rand] = v[j_rand]  # Ensure at least one gene from mutant
                
                # Bounds
                child = np.clip(child, lo, hi)
                
                children_X[i] = child
                children_F[i] = problem(child)
                
            used += pop_size
            
            # Combine Parents and Offspring
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])
            
            # Select New Population (NSGA-II style selection from 2N)
            ranks_comb = non_dominated_sort(F_combined)
            distances_comb = crowding_distance(F_combined, ranks_comb)
            
            # Sort by rank asc, distance desc
            sort_indices = np.lexsort((-distances_comb, ranks_comb))
            
            X = X_combined[sort_indices[:pop_size]]
            F = F_combined[sort_indices[:pop_size]]

        # If we have remaining budget, use it for a final local refinement or just return
        # With pop_size=20 and budget=200, we use exactly 200.
        # If budget is not a multiple, we might have leftovers.
        if used < self.budget:
             # Use remaining budget for random perturbation of best points?
             # Or just ignore for simplicity as 200 is divisible by 20.
             pass

        return X
# EVOLVE-BLOCK-END
