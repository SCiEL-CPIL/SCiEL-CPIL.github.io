import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20  # Population size
        eta_c = 20.0
        eta_m = 20.0
        
        # 1. Initialize Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # Dominance count
            n_doms = [0] * n  # Number of solutions dominated by i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # If i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    # If j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
                        
            for i in range(n):
                if n_doms[i] == 0:
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k] != []:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
                
            # Remove last empty front
            if fronts and fronts[-1] == []:
                fronts.pop()
                
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F[i, m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_range = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                if f_range == 0:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / f_range
                    
            return dist

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            child1 = np.copy(x1)
            child2 = np.copy(x2)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x):
            x_mut = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] += delta * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used + N <= self.budget:
            # Create Offspring
            offspring_X = []
            for _ in range(N):
                # Tournament Selection
                p1_idx, p2_idx = 0, 0
                # Simple random selection for parents? Or tournament?
                # Let's use tournament on rank and crowding
                # First, we need to select from current X, F
                # We'll do this in the selection step below, but for now just random parents
                # Actually, let's do proper selection in the next step.
                # For offspring generation, we need parents from current population.
                # Let's select 2 parents using tournament.
                
                # We need to rank current population
                # To avoid recomputing ranks every time, we can just pick random parents 
                # or implement a simple tournament.
                # Given N=20, recomputing is cheap.
                
                # Let's implement a helper to select 2 parents
                def select_two_parents(X, F):
                    # Calculate ranks and crowding
                    fronts = non_dominated_sort(F)
                    rank = np.zeros(len(F), dtype=int)
                    for i, front in enumerate(fronts):
                        for idx in front:
                            rank[idx] = i
                    
                    # Calculate crowding for each front
                    crowding = np.zeros(len(F))
                    for front in fronts:
                        cd = crowding_distance(F, front)
                        for idx in front:
                            crowding[idx] = cd[idx]
                    
                    # Tournament
                    def tournament():
                        idx1 = np.random.randint(len(F))
                        idx2 = np.random.randint(len(F))
                        if rank[idx1] != rank[idx2]:
                            return idx1 if rank[idx1] < rank[idx2] else idx2
                        else:
                            return idx1 if crowding[idx1] > crowding[idx2] else idx2
                    
                    p1 = tournament()
                    p2 = tournament()
                    return X[p1], X[p2]

                p1, p2 = select_two_parents(X, F)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            # We generated 2*N offspring, but we only have budget for N?
            # Standard NSGA-II generates N offspring to replace N parents.
            # My loop above generates 2*N.
            # Let's adjust: Generate N offspring.
            # I will rewrite the offspring generation to produce exactly N.
            
            # Reset offspring list
            offspring_X = []
            for _ in range(N):
                p1, p2 = select_two_parents(X, F)
                c1, _ = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                offspring_X.append(c1)
            
            # Evaluate Offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine Parents and Offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Select N best from combined
            fronts = non_dominated_sort(combined_F)
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to select from this front based on crowding
                    cd = crowding_distance(combined_F, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    remaining = N - len(selected_indices)
                    selected_indices.extend(sorted_front[:remaining])
                    break
            
            # Update Population
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]
            
        return X
