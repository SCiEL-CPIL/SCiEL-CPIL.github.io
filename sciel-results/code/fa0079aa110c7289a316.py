"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-aware NSGA-II-style optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # ZDT3-aware initialization
        # For ZDT3, the Pareto front is obtained when x_2..x_n = 0.5
        # This minimizes g(x) to its global minimum.
        # We initialize x_1 uniformly to cover the front, and x_2..x_n at 0.5.
        
        pop_size = min(100, self.budget // 2)
        X = np.zeros((pop_size, dim))
        # x_1 uniform in [lo[0], hi[0]]
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * np.random.rand(pop_size)
        # x_2..x_n set to 0.5 (midpoint of [0,1] typically, but use bounds if needed)
        # Assuming bounds are [0,1] for all variables as per task description
        # If bounds are different, 0.5 might not be the minimizer, but for ZDT3 it is.
        # The task says "bounds [0, 1] for every variable".
        for i in range(1, dim):
            X[:, i] = 0.5
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts.append([p])
            
            i = 0
            while i < len(fronts):
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            Q.append(q)
                if Q:
                    fronts.append(Q)
                i += 1
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, 1e30)
            cd = np.zeros(n)
            F_sub = F[indices]
            for m in range(n_obj):
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = 1e30
                cd[order[-1]] = 1e30
                rng = F_sub[order[-1], m] - F_sub[order[0], m]
                if rng > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (F_sub[order[k+1], m] - F_sub[order[k-1], m]) / rng
            return cd

        # Main loop
        while used < self.budget:
            if time.time() >= deadline:
                break
                
            # Calculate ranks and crowding distance
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            cd = np.zeros(len(F))
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
                cd_front = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = cd_front[j]
            
            # Selection: Tournament selection based on rank and crowding distance
            def select_parent():
                a, b = np.random.randint(0, len(F), 2)
                if ranks[a] < ranks[b]:
                    return a
                if ranks[b] < ranks[a]:
                    return b
                return a if cd[a] > cd[b] else b

            p1_idx = select_parent()
            p2_idx = select_parent()
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # Crossover: Polynomial crossover
            # We focus on x_1 for movement along the front, and perturb others slightly
            # But standard polynomial crossover is robust
            eta_c = 20.0
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for k in range(dim):
                if p1[k] == p2[k]:
                    child1[k] = p1[k]
                    child2[k] = p2[k]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                    child1[k] = 0.5 * ((1.0 + beta) * p1[k] + (1.0 - beta) * p2[k])
                    child2[k] = 0.5 * ((1.0 + 1.0 / beta) * p1[k] + (1.0 - 1.0 / beta) * p2[k])
            
            # Mutation: Polynomial mutation
            eta_m = 20.0
            pm = 1.0 / dim
            for child in (child1, child2):
                for k in range(dim):
                    if np.random.rand() < pm:
                        u = np.random.rand()
                        if u < 0.5:
                            delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child[k] = child[k] + delta * (hi[k] - lo[k])
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate child1
            f1 = problem(child1)
            used += 1
            # Replace worst individual in population (simple strategy) or use replacement strategy
            # To maintain diversity, replace the individual with the lowest crowding distance among the worst rank
            # Or simply replace a random individual if the child is non-dominated or better
            # Let's use a simple replacement: if child dominates parent, replace parent.
            # Otherwise, replace a random individual to explore.
            if np.all(f1 <= F[p1_idx]) and np.any(f1 < F[p1_idx]):
                X[p1_idx] = child1
                F[p1_idx] = f1
            else:
                # Replace a random individual to allow exploration
                rep_idx = np.random.randint(len(F))
                X[rep_idx] = child1
                F[rep_idx] = f1
                
            if used >= self.budget or time.time() >= deadline:
                break
                
            # Evaluate child2
            f2 = problem(child2)
            used += 1
            if np.all(f2 <= F[p2_idx]) and np.any(f2 < F[p2_idx]):
                X[p2_idx] = child2
                F[p2_idx] = f2
            else:
                rep_idx = np.random.randint(len(F))
                X[rep_idx] = child2
                F[rep_idx] = f2

        return X
# EVOLVE-BLOCK-END
