import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 50 # Population size
        # Check if we can afford at least one generation
        if self.budget < N:
            # If budget is too small, just evaluate random points
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return self._get_non_dominated(F)

        # 1. Initialize Population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II Parameters
        eta_c = 20.0 # Crossover distribution index
        eta_m = 20.0 # Mutation distribution index
        p_m = 1.0 / self.dim # Probability of mutation per variable
        
        # Helper Functions
        
        def non_dominated_sort(f):
            """
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(f)
            S = [[] for _ in range(n)] # Dominates
            n_dom = np.zeros(n, dtype=int) # Number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                        S[i].append(j)
                    elif np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                        n_dom[i] += 1
            
            fronts = []
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            if current_front:
                fronts.append(current_front)
                
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
                if current_front:
                    fronts.append(current_front)
            
            return fronts, rank

        def crowding_distance(f, indices):
            """
            Calculates crowding distance for a set of indices in f.
            Returns a dict mapping index to distance.
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            m = len(indices)
            for k in range(self.n_obj):
                # Sort indices by objective k
                sorted_indices = sorted(indices, key=lambda i: f[i][k])
                # Assign infinite distance to boundary points
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                # Normalize objective range
                f_min = f[sorted_indices[0]][k]
                f_max = f[sorted_indices[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_indices[i]] += (f[sorted_indices[i+1]][k] - f[sorted_indices[i-1]][k]) / f_range
            
            return dist

        def sbx_crossover(x1, x2):
            """
            Simulated Binary Crossover.
            """
            x1 = np.array(x1)
            x2 = np.array(x2)
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x1)
            
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    # Swap x1, x2 if necessary so x1 < x2 for calculation consistency, 
                    # though SBX formula is symmetric if handled correctly.
                    # Standard SBX:
                    if x1[i] == x2[i]:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
                        continue
                    
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            return c1, c2

        def polynomial_mutation(x):
            """
            Polynomial Mutation.
            """
            x = np.array(x)
            for i in range(self.dim):
                if np.random.random() < p_m:
                    u = np.random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    x[i] += delta_q * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        def tournament_select(pop_indices, ranks, crowding):
            """
            Tournament selection based on rank and crowding distance.
            """
            k = 2
            candidates = np.random.choice(pop_indices, size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                # Compare ranks
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best]:
                    # Compare crowding distance
                    if crowding[c] > crowding[best]:
                        best = c
            return best

        # Main Evolution Loop
        while used < self.budget:
            # Calculate fronts and crowding for current population
            fronts, ranks = non_dominated_sort(F)
            
            # Calculate crowding distance for the whole population 
            # (NSGA-II usually calculates it per front, but for selection 
            # we need it for all individuals in the population)
            # Actually, standard NSGA-II selection uses rank first.
            # We can compute crowding for all points in the current population.
            all_indices = list(range(N))
            crowding = crowding_distance(F, all_indices)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # Number of offspring to generate in this generation
            # We can generate up to N offspring, but we must respect the budget.
            # We generate pairs.
            max_offspring = min(N, self.budget - used)
            
            for _ in range(max_offspring // 2):
                # Select parents
                p1_idx = tournament_select(range(N), ranks, crowding)
                p2_idx = tournament_select(range(N), ranks, crowding)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            # P_t + Q_t -> R_t
            R_X = np.vstack([X, offspring_X])
            R_F = np.vstack([F, offspring_F])
            R_size = len(R_X)
            
            # Non-dominated sorting of R_t
            R_fronts, R_ranks = non_dominated_sort(R_F)
            
            # Select next population P_{t+1}
            new_X = []
            new_F = []
            
            # Calculate crowding distance for each front in R_t
            # We need to assign crowding distance to all individuals in R_t
            # Standard algorithm:
            # 1. Sort R_t into fronts F_1, F_2, ...
            # 2. Add fronts to P_{t+1} until capacity is reached.
            # 3. For the last front that fits partially, sort by crowding distance and take the best.
            
            # First, calculate crowding for all individuals in R_t?
            # No, we only need it for the front where we have to truncate.
            # However, tournament selection in the next iteration will need crowding for the new population.
            # So we can calculate crowding for the new population after selection.
            
            current_size = 0
            for i, front in enumerate(R_fronts):
                if current_size + len(front) <= N:
                    # Add entire front
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    current_size += len(front)
                else:
                    # Need to select from this front
                    remaining = N - current_size
                    if remaining > 0:
                        # Calculate crowding distance for this specific front
                        front_crowding = crowding_distance(R_F, front)
                        # Sort front by crowding distance (descending)
                        sorted_front = sorted(front, key=lambda idx: front_crowding[idx], reverse=True)
                        # Take the best 'remaining' individuals
                        for idx in sorted_front[:remaining]:
                            new_X.append(R_X[idx])
                            new_F.append(R_F[idx])
                    break
            
            # Update population
            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
            else:
                break

        # Final Non-Dominated Sort for return
        return self._get_non_dominated(F)

    def _get_non_dominated(self, F):
        """
        Returns the indices of non-dominated solutions in F.
        """
        n = len(F)
        if n == 0:
            return np.array([])
        
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        
        if not non_dominated:
             # Fallback if all are dominated (should not happen with at least 1 point)
             return np.array([0])
             
        return np.array(non_dominated)
