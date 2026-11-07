"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Evolutionary multi-objective search with NSGA-II-style selection and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 295.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a fixed seed for internal RNG to ensure reproducibility 
        # and independence from the global np.random state, 
        # unless we want to leverage the evaluator's seed. 
        # The evaluator calls np.random.seed(seed). 
        # If we use np.random directly, we get different runs for seed 0 and 1.
        # If we use default_rng(0), we get the same run.
        # The task says "For seeds 0 and 1 ... creates Optimizer ... and calls it once".
        # "Scoring ... is the mean over the two seeds".
        # If the algorithm is deterministic (same seed), the score is the same for both.
        # If it uses global np.random, it varies.
        # Usually, using the global state is preferred to explore different regions.
        rng = np.random
        
        pop_size = 20
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Simple non-dominated sorting. Returns list of lists of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = [0] * n  # n_d[i] = number of solutions dominating i
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                    elif dominates(F[q], F[p]):
                        n_d[p] += 1
            
            front = []
            for p in range(n):
                if n_d[p] == 0:
                    front.append(p)
            
            i = 0
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                front = next_front
                i += 1
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices in the same front."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            F_sub = F[indices]
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(indices, key=lambda i: F_sub[i, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                
                if f_max == f_min:
                    continue
                
                for i in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / (f_max - f_min)
            
            return dist

        # Main Loop
        while used < self.budget:
            if time.time() - start_time > max_time:
                break
            
            # Select parents from the best front
            fronts = non_dominated_sort(F)
            best_front = fronts[0]
            
            # Pick a random parent from the best front
            parent_idx = best_front[rng.integers(len(best_front))]
            parent = X[parent_idx]
            
            # Mutate
            # Adaptive mutation step size
            step = 0.1 * (hi - lo)
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add child to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])
            
            # Select best pop_size individuals
            # Combine and sort
            all_fronts = non_dominated_sort(F)
            
            selected_X = []
            selected_F = []
            
            for front in all_fronts:
                if len(selected_X) + len(front) <= pop_size:
                    # Take all
                    selected_X.extend(X[front])
                    selected_F.extend(F[front])
                else:
                    # Need to truncate
                    needed = pop_size - len(selected_X)
                    cd = crowding_distance(F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected_X.extend(X[sorted_front[:needed]])
                    selected_F.extend(F[sorted_front[:needed]])
                    break
            
            X = np.array(selected_X)
            F = np.array(selected_F)
            
        return X
# EVOLVE-BLOCK-END
