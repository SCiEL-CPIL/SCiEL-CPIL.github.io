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
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size and generations
        # Budget is 200. Let's use pop_size=20.
        # Initial eval: 20.
        # Remaining: 180.
        # Generations: 180 / 20 = 9 generations of offspring.
        # Total evaluations: 20 + 9*20 = 200.
        pop_size = 20
        n_gen = (self.budget - pop_size) // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        # SBX params
        eta_c = 20.0
        # Polynomial mutation params
        eta_m = 20.0
        p_m = 1.0 / self.dim
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    # F[i] <= F[j] and F[i] != F[j]
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                        
            # First front
            for i in range(n):
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
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                # Sort by k-th objective
                sorted_idx = np.argsort(F[front_indices, k])
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                # Assign infinite distance to boundary points
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                # Normalize range
                f_min = F[front_indices[sorted_idx[0]], k]
                f_max = F[front_indices[sorted_idx[-1]], k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx_curr = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    
                    dist[idx_curr] += (F[front_indices[idx_next], k] - F[front_indices[idx_prev], k]) / f_range
            
            return dist

        def select_parents(F, X):
            fronts = fast_non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    front_cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = front_cd[i]
            
            # Tournament selection (size 2)
            parents = []
            for _ in range(pop_size):
                i1 = np.random.randint(len(F))
                i2 = np.random.randint(len(F))
                
                # Compare rank first, then crowding distance
                if rank[i1] < rank[i2]:
                    parents.append(i1)
                elif rank[i2] < rank[i1]:
                    parents.append(i2)
                else:
                    if cd[i1] > cd[i2]:
                        parents.append(i1)
                    else:
                        parents.append(i2)
            return parents

        def sbx_crossover(x1, x2):
            if np.random.rand() < 0.9: # Crossover probability
                u = np.random.random(self.dim)
                beta = np.zeros(self.dim)
                for i in range(self.dim):
                    if u[i] <= 0.5:
                        beta[i] = (2 * u[i]) ** (1.0 / (eta_c + 1))
                    else:
                        beta[i] = (1.0 / (2 * (1.0 - u[i]))) ** (1.0 / (eta_c + 1))
                
                c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                
                # Clip to bounds
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x):
            u = np.random.random(self.dim)
            delta = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < p_m:
                    if u[i] < 0.5:
                        delta[i] = (2 * u[i]) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta[i] = 1 - (2 * (1 - u[i])) ** (1.0 / (eta_m + 1))
            
            x_new = x + delta * (hi - lo)
            return np.clip(x_new, lo, hi)

        # Main loop
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            
            # Select parents
            parent_indices = select_parents(F, X)
            
            # Create offspring
            offspring_X = []
            for _ in range(pop_size // 2):
                p1_idx = parent_indices[2*_]
                p2_idx = parent_indices[2*_ + 1]
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    if remaining <= 0:
                        break
                    
                    # Calculate crowding distance for this front
                    front_cd = crowding_distance(combined_F, front)
                    
                    # Sort by crowding distance (descending)
                    sorted_indices = np.argsort(-front_cd)
                    
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
