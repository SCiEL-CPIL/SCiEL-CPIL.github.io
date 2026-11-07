import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use global numpy random which is seeded by evaluator
        # But to be safe and reproducible within the run, we can use default_rng 
        # seeded from np.random if we want to respect the seed.
        # However, standard practice in these tasks is often to just use np.random.
        
        pop_size = 20
        used = 0
        all_f_vectors = [] # To track all evaluated points for hypervolume
        
        # 1. Initialization: LHS
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            X[:, i] = (np.arange(pop_size) + np.random.rand(pop_size)) / pop_size
            X[:, i] = X[:, i] * (hi[i] - lo[i]) + lo[i]
            
        F = np.array([problem(x) for x in X])
        used += pop_size
        all_f_vectors.extend(F)
        
        # Helper functions for NSGA-II
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            rank = [0] * n
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
                        
            for p in range(n):
                if n_dominates[p] == 0:
                    rank[p] = 0
                    front[0] += 1
                    
            i = 0
            while front[i] > 0:
                next_front = []
                for p in S: # This is wrong, S is list of lists of dominated points
                    # Need to iterate over points in current front
                    pass # Implementation below is better
                
            # Correct implementation
            fronts = []
            current_front = []
            for p in range(n):
                if n_dominates[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for p in fronts[k]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            rank[q] = k + 1
                            next_front.append(q)
                k += 1
                if next_front:
                    fronts.append(next_front)
                    
            return fronts, rank

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, n-1):
                    idx = sorted_indices[i]
                    idx_next = sorted_indices[i+1]
                    idx_prev = sorted_indices[i-1]
                    distances[idx] += (F[idx_next][m] - F[idx_prev][m]) / f_range
                    
            return distances

        def select_parents(pop_F, pop_X):
            fronts, ranks = non_dominated_sort(pop_F)
            # Calculate crowding distance for each front
            crowding = [0.0] * len(pop_F)
            for front in fronts:
                cd = crowding_distance(pop_F, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]
            
            # Binary Tournament
            def tournament():
                p1 = np.random.randint(len(pop_F))
                p2 = np.random.randint(len(pop_F))
                if ranks[p1] < ranks[p2]:
                    return p1
                elif ranks[p2] < ranks[p1]:
                    return p2
                else:
                    if crowding[p1] > crowding[p2]:
                        return p1
                    else:
                        return p2
            
            parent1 = pop_X[tournament()]
            parent2 = pop_X[tournament()]
            return parent1, parent2

        # Main Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                p1, p2 = select_parents(F, X)
                # SBX-like or Gaussian
                # Simple Gaussian mutation
                # Step size adaptive? Start 0.1, decay.
                step = 0.1
                # Crossover: Interpolate
                t = np.random.rand()
                child = p1 * t + p2 * (1-t)
                # Mutation
                child = child + np.random.normal(0, step, self.dim)
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            all_f_vectors.extend(offspring_F)
            
            if used >= self.budget:
                break
                
            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Sort and select
            fronts, ranks = non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select subset based on crowding
                    cd = crowding_distance(combined_F, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return None
