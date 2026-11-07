"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using a Differential Evolution (DE) approach.
    DE is robust for deceptive landscapes like WFG9.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # DE usually works well with NP = 10-20 * D. Here D=12, so 120-240.
        # Budget is 200. Let's try a smaller population to allow more generations?
        # Or a larger population for better coverage?
        # With budget 200, if NP=50, we get 4 generations. If NP=100, 2 generations.
        # Let's try NP=50 to allow more iterations.
        pop_size = 50
        if self.budget < pop_size:
            pop_size = self.budget
            
        X = (lo + (hi - lo) * np.random.random((pop_size, self.dim)))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # DE parameters
        F_cr = 0.8  # Crossover rate
        F_scale = 0.5  # Scaling factor
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return ranks

        def crowding_distance(F, ranks, n):
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    min_val, max_val = np.min(vals), np.max(vals)
                    if max_val - min_val < 1e-10:
                        cd[idx] = np.inf
                        continue
                    order = np.argsort(vals)
                    sorted_idx = idx[order]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (vals[order[i+1]] - vals[order[i-1]]) / (max_val - min_val)
            return cd

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate one trial vector per individual in population
            # We can only afford to generate (budget - used) trial vectors total.
            # Standard DE generates NP trial vectors per generation.
            # If we have budget left for a full generation, do it.
            # Otherwise, do partial generation?
            # Let's just do one trial vector at a time to be safe with budget.
            
            # Select target
            i = np.random.randint(pop_size)
            
            # Select distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # DE/rand/1 mutation
            v = X[r1] + F_scale * (X[r2] - X[r3])
            
            # Binary crossover
            u = np.zeros_like(X[i])
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.random() < F_cr or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = X[i][j]
            
            # Clip to bounds
            u = np.clip(u, lo, hi)
            
            # Evaluate
            f_u = problem(u)
            used += 1
            
            # Selection
            if dominates(f_u, F[i]):
                X[i] = u
                F[i] = f_u
            # If F[i] dominates f_u, keep F[i]
            # If they are non-dominated, we could keep both? 
            # Standard DE replaces if trial is better or equal in some sense?
            # Usually, if trial is not dominated by target, it replaces target?
            # Or we can use a simple rule: replace if trial is non-dominated by target.
            elif not dominates(F[i], f_u):
                # Non-dominated
                # Keep the one with better crowding distance?
                # For simplicity, let's keep the trial if it's non-dominated.
                # This promotes diversity.
                X[i] = u
                F[i] = f_u

        return X
# EVOLVE-BLOCK-END
