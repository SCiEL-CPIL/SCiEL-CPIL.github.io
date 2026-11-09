import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        all_X = []
        all_F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        # Population size for NSGA-II
        pop_size = 40
        n_generations = (self.budget - pop_size) // pop_size
        if n_generations < 1:
            n_generations = 1
            pop_size = min(self.budget, pop_size)

        # Initialize population with Latin Hypercube Sampling
        pop = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            edges = np.linspace(0, 1, pop_size + 1)
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points_in_bins)
            pop[:, d] = points_in_bins
        
        pop = lo + pop * span
        
        # Evaluate initial population
        for i in range(pop_size):
            if n_evals >= self.budget or time.time() > deadline:
                break
            fx = evaluate(pop[i])
            all_X.append(pop[i].copy())
            all_F.append(fx)
        
        if n_evals >= self.budget or time.time() > deadline:
            return np.array(all_X)
        
        # NSGA-II helper functions
        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = [0] * n  # number of solutions dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                
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
                k += 1
                fronts.append(next_front)
            
            fronts.pop()  # Remove last empty front
            return fronts
        
        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            F_front = F[front_indices]
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F_front[:, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F_front[sorted_idx[i+1], m] - F_front[sorted_idx[i-1], m]) / f_range
            
            return dist
        
        # Main NSGA-II loop
        current_pop_idx = list(range(pop_size))  # Indices into all_X/all_F for current population
        
        for gen in range(n_generations):
            if n_evals >= self.budget or time.time() > deadline:
                break
            
            # Get current population objectives
            F_pop = np.array([all_F[i] for i in current_pop_idx])
            
            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F_pop)
            
            # Select parents using tournament selection based on rank and crowding
            # Assign ranks
            rank = [0] * pop_size
            for front_num, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_num
            
            # Calculate crowding distance for each front
            crowd_dist = [0.0] * pop_size
            for front in fronts:
                cd = crowding_distance(F_pop, front)
                for i, idx in enumerate(front):
                    crowd_dist[idx] = cd[i]
            
            # Binary tournament selection
            def tournament_select():
                i = np.random.randint(pop_size)
                j = np.random.randint(pop_size)
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    return i if crowd_dist[i] > crowd_dist[j] else j
            
            # Generate offspring
            offspring = []
            offspring_F = []
            
            n_offspring = pop_size
            for _ in range(n_offspring):
                if n_evals >= self.budget or time.time() > deadline:
                    break
                
                # Select parents
                p1 = tournament_select()
                p2 = tournament_select()
                
                # SBX crossover
                x1 = all_X[current_pop_idx[p1]].copy()
                x2 = all_X[current_pop_idx[p2]].copy()
                
                if np.random.rand() < 0.9:  # Crossover probability
                    # Simulated Binary Crossover
                    eta = 20.0
                    children = []
                    for d in range(self.dim):
                        if x1[d] == x2[d]:
                            children.append([x1[d], x2[d]])
                            continue
                        
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        c1 = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        c2 = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                        
                        # Clip to bounds
                        c1 = np.clip(c1, lo[d], hi[d])
                        c2 = np.clip(c2, lo[d], hi[d])
                        children.append([c1, c2])
                    
                    child1 = np.array([c[0] for c in children])
                    child2 = np.array([c[1] for c in children])
                else:
                    child1 = x1.copy()
                    child2 = x2.copy()
                
                # Polynomial mutation
                eta_mut = 20.0
                for d in range(self.dim):
                    if np.random.rand() < 1.0 / self.dim:  # Mutation probability
                        u = np.random.rand()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (eta_mut + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_mut + 1))
                        
                        span_d = hi[d] - lo[d]
                        child1[d] = np.clip(child1[d] + delta * span_d, lo[d], hi[d])
                    
                    if np.random.rand() < 1.0 / self.dim:
                        u = np.random.rand()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (eta_mut + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_mut + 1))
                        
                        span_d = hi[d] - lo[d]
                        child2[d] = np.clip(child2[d] + delta * span_d, lo[d], hi[d])
                
                # Evaluate offspring
                f1 = evaluate(child1)
                f2 = evaluate(child2)
                
                offspring.append(child1)
                offspring_F.append(f1)
                offspring.append(child2)
                offspring_F.append(f2)
            
            if not offspring:
                break
            
            # Combine parent and offspring populations
            combined_X = [all_X[i].copy() for i in current_pop_idx] + offspring
            combined_F = np.array([all_F[i] for i in current_pop_idx] + offspring_F)
            
            # Non-dominated sort on combined population
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_pop_idx = []
            for front in combined_fronts:
                if len(next_pop_idx) + len(front) <= pop_size:
                    next_pop_idx.extend(front)
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(next_pop_idx)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    next_pop_idx.extend(sorted_front[:remaining])
                    break
            
            # Update current population
            # Map from combined indices to all_X/all_F indices
            # First pop_size entries are from current_pop_idx, rest are new
            new_current_pop_idx = []
            for idx in next_pop_idx:
                if idx < len(current_pop_idx):
                    new_current_pop_idx.append(current_pop_idx[idx])
                else:
                    # It's a new offspring, add to all_X and all_F
                    new_X = combined_X[idx]
                    new_F = combined_F[idx]
                    all_X.append(new_X)
                    all_F.append(new_F)
                    new_current_pop_idx.append(len(all_X) - 1)
            
            current_pop_idx = new_current_pop_idx
        
        return np.array(all_X)
