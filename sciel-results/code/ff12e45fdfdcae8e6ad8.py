"""Multi-objective optimizer for RE34 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size
        # For 5 dims, 30-50 is reasonable. We have 200 budget.
        # Let's use 40 to allow for ~4-5 generations of (P+P) selection.
        pop_size = 40

        # Initialize population randomly
        X = lo + span * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions for NSGA-II

        def fast_non_dominated_sort(F):
            """
            Perform fast non-dominated sorting.
            Returns a list of lists of indices, where each inner list is a front.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
            rank = np.zeros(n, dtype=int)  # rank[i] = front index (0-based)
            
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        if n_dom[i] == 0:
                            rank[i] = 0
                            current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a front.
            Returns a dictionary of index -> distance.
            """
            n = len(front_indices)
            if n <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            
            for m in range(self.n_obj):
                # Sort front by objective m
                sorted_front = sorted(front_indices, key=lambda i: F[i][m])
                # Assign inf to boundary points
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]
                if f_max - f_min == 0:
                    continue
                
                for i in range(1, n - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i + 1]][m] - F[sorted_front[i - 1]][m]) / (f_max - f_min)
            
            return dist

        def tournament_select(rank, dist, X, F, pop_size):
            """
            Binary tournament selection.
            """
            selected = []
            for _ in range(pop_size):
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                
                if rank[i1] < rank[i2]:
                    winner = i1
                elif rank[i1] > rank[i2]:
                    winner = i2
                else:
                    if dist[i1] > dist[i2]:
                        winner = i1
                    else:
                        winner = i2
                selected.append(winner)
            return selected

        def sbx_crossover(x1, x2, eta=20.0):
            """
            Simulated Binary Crossover.
            """
            x = np.zeros_like(x1)
            for j in range(dim):
                if np.random.rand() < 0.5:
                    # Generate two children, we only need one here? 
                    # Standard NSGA-II usually generates 2 offspring from 2 parents.
                    # Here we will generate 2 and return both.
                    pass
            
            # We will implement a version that returns two children
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x1)
            
            for j in range(dim):
                if np.random.rand() < 0.5:
                    c1[j] = x1[j]
                    c2[j] = x2[j]
                else:
                    if abs(x1[j] - x2[j]) < 1e-14:
                        c1[j] = x1[j]
                        c2[j] = x2[j]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        
                        c1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        c2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                        
                        # Clip to bounds
                        c1[j] = np.clip(c1[j], lo[j], hi[j])
                        c2[j] = np.clip(c2[j], lo[j], hi[j])
            return c1, c2

        def polynomial_mutation(x, eta=20.0):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for j in range(dim):
                if np.random.rand() < 1.0 / dim:
                    u = np.random.rand()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    range_val = hi[j] - lo[j]
                    x_mut[j] = x_mut[j] + delta_q * range_val
                    x_mut[j] = np.clip(x_mut[j], lo[j], hi[j])
            return x_mut

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Evaluate current population if not done (initial eval was done)
            # In NSGA-II, we maintain a population of size N.
            # We create offspring of size N, combine to 2N, select N.
            
            # 1. Non-dominated sort
            fronts, rank = fast_non_dominated_sort(F)
            
            # 2. Crowding distance
            dist = {}
            for front in fronts:
                front_dist = crowding_distance(F, front)
                dist.update(front_dist)
            
            # 3. Create offspring
            offspring_X = []
            offspring_F = []
            
            # We need to create pop_size offspring
            # Each offspring takes 1 evaluation (if we do 1 crossover + mutation per offspring? 
            # Usually SBX produces 2 children. If we produce 2 children per pair, we need pop_size/2 pairs.
            # Let's produce 2 children per pair to be efficient.
            
            n_pairs = pop_size // 2
            if pop_size % 2 != 0:
                n_pairs += 1 # Handle odd, though we set 40
            
            for _ in range(n_pairs):
                # Select 2 parents
                parents = tournament_select(rank, dist, X, F, 2)
                p1, p2 = parents
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                if used < budget:
                    offspring_X.append(c1)
                    offspring_F.append(f1)
                    used += 1
                else:
                    break
                
                if used < budget:
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                    used += 1
                else:
                    break
            
            if not offspring_X:
                break
            
            # Combine parent and offspring
            # Note: offspring_X might be smaller than pop_size if budget ran out exactly, 
            # but we check budget in loop.
            # If we broke early, we might have fewer offspring. 
            # For simplicity, if we don't have enough offspring to make 2N, we just use what we have?
            # No, NSGA-II requires 2N for proper selection.
            # If budget is tight, we might stop.
            
            # Let's assume we have enough budget to complete a generation or we stop.
            # If len(offspring_X) < pop_size, we can't do standard 2N selection properly.
            # We will only proceed if we have a full population of offspring.
            if len(offspring_X) < pop_size:
                break
            
            X_child = np.array(offspring_X[:pop_size])
            F_child = np.array(offspring_F[:pop_size])
            
            # Combine
            X_combined = np.vstack([X, X_child])
            F_combined = np.vstack([F, F_child])
            
            # Select next generation
            fronts, rank = fast_non_dominated_sort(F_combined)
            
            dist = {}
            for front in fronts:
                front_dist = crowding_distance(F_combined, front)
                dist.update(front_dist)
            
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(front_sorted[:needed])
                    break
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
            # Optional: check for stagnation? 
            # For now, rely on NSGA-II diversity.

        return X
# EVOLVE-BLOCK-END
