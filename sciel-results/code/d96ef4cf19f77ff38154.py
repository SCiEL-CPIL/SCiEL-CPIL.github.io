"""Multi-objective optimizer for RE34 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        # 200 budget. 50 init + 50*3 generations = 200.
        pop_size = 50
        
        # Initialize population using Latin Hypercube for better spread
        X = np.empty((pop_size, dim))
        for d in range(dim):
            # Stratified sampling
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        X = X * (hi - lo) + lo

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evaluated = pop_size

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            """
            Returns ranks for each individual.
            F is (N, n_obj). Lower is better.
            """
            N = len(F)
            ranks = np.zeros(N, dtype=int)
            S = [[] for _ in range(N)]  # S[i] = indices j dominated by i
            n = [0] * N  # n[i] = number of individuals dominating i
            
            # Compute dominance relationships
            for i in range(N):
                for j in range(i + 1, N):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n[i] += 1
            
            # Assign ranks
            current_front = []
            for i in range(N):
                if n[i] == 0:
                    ranks[i] = 0
                    current_front.append(i)
            
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            ranks[j] = rank + 1
                            next_front.append(j)
                current_front = next_front
                rank += 1
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            """
            Returns crowding distance for each individual.
            """
            N = len(F)
            distances = np.zeros(N)
            unique_ranks = np.unique(ranks)
            
            for r in unique_ranks:
                front_indices = np.where(ranks == r)[0]
                n_front = len(front_indices)
                
                if n_front <= 2:
                    # Edge cases
                    for idx in front_indices:
                        distances[idx] = np.inf
                    continue
                
                # Sort front by each objective
                for k in range(self.n_obj):
                    sorted_indices = front_indices[np.argsort(F[front_indices, k])]
                    # Assign inf to extremes
                    distances[sorted_indices[0]] = np.inf
                    distances[sorted_indices[-1]] = np.inf
                    
                    # Normalize objective range for this front
                    f_min = F[sorted_indices[0], k]
                    f_max = F[sorted_indices[-1], k]
                    f_range = f_max - f_min
                    if f_range == 0:
                        continue
                    
                    # Calculate distances for interior points
                    for i in range(1, n_front - 1):
                        dist = (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / f_range
                        distances[sorted_indices[i]] += dist
                        
            return distances

        # Helper: Tournament selection
        def tournament_select(pop_X, pop_F, ranks, distances, k=2):
            """
            Select one individual from population using tournament.
            """
            N = len(pop_X)
            indices = np.random.choice(N, size=k, replace=False)
            
            best_idx = indices[0]
            for idx in indices[1:]:
                # Compare rank first
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx]:
                    # Then compare crowding distance (higher is better)
                    if distances[idx] > distances[best_idx]:
                        best_idx = idx
            return pop_X[best_idx].copy()

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, prob=0.9, eta=20.0):
            """
            Simulated Binary Crossover.
            """
            if np.random.rand() > prob:
                return x1.copy(), x2.copy()
            
            x3 = np.zeros_like(x1)
            x4 = np.zeros_like(x2)
            
            for i in range(len(x1)):
                if x1[i] == x2[i]:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    # Clip to bounds
                    x3[i] = np.clip(x3[i], lo[i], hi[i])
                    x4[i] = np.clip(x4[i], lo[i], hi[i])
            
            return x3, x4

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, prob=1.0/dim, eta=20.0):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    yl = (x[i] - lo[i]) / (hi[i] - lo[i])
                    yr = (hi[i] - x[i]) / (hi[i] - lo[i])
                    
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                        x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while evaluated + pop_size <= self.budget:
            if time.time() > deadline:
                break
            
            # Calculate ranks and distances for current population
            ranks = non_dominated_sort(F)
            distances = crowding_distance(F, ranks)
            
            # Generate offspring
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size):
                # Select parents
                p1 = tournament_select(X, F, ranks, distances)
                p2 = tournament_select(X, F, ranks, distances)
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                
                # We only have space for pop_size offspring in our budget logic (50).
                # So we just take one child per pair? Or do we do 25 pairs?
                # The loop runs pop_size times. So we generate 1 offspring per iteration?
                # That would mean 50 offspring.
                # Let's just take c1.
                offspring_X[i] = c1
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            evaluated += pop_size
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection for next generation
            combined_ranks = non_dominated_sort(combined_F)
            combined_distances = crowding_distance(combined_F, combined_ranks)
            
            # Select best pop_size individuals
            # We need to select based on rank and distance
            # Create a list of (rank, -distance, index) and sort
            selection_indices = []
            
            # Sort indices by rank, then by -distance (descending distance)
            # np.lexsort sorts by last key first.
            # We want ascending rank, then descending distance.
            # So we sort by (rank, -distance).
            sort_keys = np.lexsort((-combined_distances, combined_ranks))
            
            # Take the first pop_size indices
            selected_indices = sort_keys[:pop_size]
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
