import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # NSGA-II Parameters
        pop_size = 20
        n_gen = self.budget // pop_size
        if n_gen == 0:
            n_gen = 1
            pop_size = self.budget
            
        # Initialize Population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = np.zeros(n, dtype=int)  # Dominance count
            fronts = [[]]  # List of fronts, each front is a list of indices
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                        
                if n_d[p] == 0:
                    fronts[0].append(p)
                    
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            Q.append(q)
                i += 1
                if Q:
                    fronts.append(Q)
                else:
                    break
                    
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            m = len(front_indices)
            cd = np.zeros(m)
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_idx = np.argsort([F[i][k] for i in front_indices])
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                # Calculate range
                min_val = F[sorted_front[0]][k]
                max_val = F[sorted_front[-1]][k]
                if max_val - min_val == 0:
                    continue
                
                cd[0] = float('inf')
                cd[-1] = float('inf')
                
                for i in range(1, m - 1):
                    cd[i] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / (max_val - min_val)
                    
            return cd

        # Main Loop
        for gen in range(n_gen):
            # Generate Offspring
            Q = np.zeros_like(X)
            for i in range(pop_size):
                # Select parents using Tournament Selection
                # We need to rank the current population
                fronts = fast_non_dominated_sort(F)
                
                # Assign ranks
                ranks = np.zeros(pop_size, dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = rank
                
                # Calculate crowding distances for each front
                cd = np.zeros(pop_size)
                for front in fronts:
                    if len(front) > 2:
                        cd_vals = crowding_distance(F, front)
                        for j, idx in enumerate(front):
                            cd[idx] = cd_vals[j]
                    else:
                        for idx in front:
                            cd[idx] = float('inf')
                
                # Tournament Selection
                def tournament():
                    idx1 = np.random.randint(pop_size)
                    idx2 = np.random.randint(pop_size)
                    if ranks[idx1] < ranks[idx2]:
                        return idx1
                    elif ranks[idx2] < ranks[idx1]:
                        return idx2
                    else:
                        if cd[idx1] > cd[idx2]:
                            return idx1
                        else:
                            return idx2
                
                p1 = tournament()
                p2 = tournament()
                
                # Crossover (SBX)
                if np.random.rand() < 0.9:
                    # Simple Blend Crossover for simplicity and robustness
                    alpha = np.random.rand(self.dim)
                    child = alpha * X[p1] + (1 - alpha) * X[p2]
                    # Clip
                    child = np.clip(child, lo, hi)
                else:
                    child = X[p1].copy()
                
                # Mutation (Polynomial or Gaussian)
                # Use Gaussian mutation with adaptive rate
                if np.random.rand() < 0.9:
                    # Scale mutation by distance to bounds
                    dist = (hi - lo)
                    noise = np.random.normal(0, 1, self.dim) * 0.1 * dist
                    child = child + noise
                    child = np.clip(child, lo, hi)
                
                Q[i] = child
            
            QF = np.array([problem(x) for x in Q])
            used += pop_size
            
            # Combine and Select
            R = np.vstack([X, Q])
            RF = np.vstack([F, QF])
            
            fronts = fast_non_dominated_sort(RF)
            
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_X.append(R[idx])
                        new_F.append(RF[idx])
                else:
                    # Need to select subset based on crowding distance
                    needed = pop_size - len(new_X)
                    cd_vals = crowding_distance(RF, front)
                    # Sort by cd descending
                    sorted_indices = np.argsort(-cd_vals)
                    for i in range(needed):
                        idx = front[sorted_indices[i]]
                        new_X.append(R[idx])
                        new_F.append(RF[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            if used >= self.budget:
                break
                
        # Return the non-dominated set of the final population
        # Or just return the population, as the evaluator records all calls.
        # The task says "Every returned objective vector is recorded".
        # So the archive is just the history of all calls.
        # We don't need to return a specific archive, just ensure we made the calls.
        # But returning the final non-dominated set is good practice.
        
        fronts = fast_non_dominated_sort(F)
        nd_set = []
        for idx in fronts[0]:
            nd_set.append(X[idx])
            
        return nd_set
