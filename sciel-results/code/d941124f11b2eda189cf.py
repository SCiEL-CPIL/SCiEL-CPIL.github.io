import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 50
        if pop_size > self.budget:
            pop_size = self.budget
            
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        eval_count = pop_size
        
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Generate Offspring
            offspring_X = []
            offspring_F = []
            
            # We will generate pop_size offspring
            # But we must respect budget
            num_offspring = min(pop_size, self.budget - eval_count)
            
            for _ in range(num_offspring):
                # Select parents using Tournament Selection
                # Tournament size 2
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                # We need ranks and crowding for selection
                # To avoid recalculating ranks for every parent selection, 
                # we should calculate them once per generation.
                # But we are inside the loop.
                # Let's restructure:
                # 1. Calculate ranks and crowding for current X, F.
                # 2. Use them for selection.
                pass
            
            # Restructure:
            # 1. Calculate selection metrics
            ranks, fronts = self.fast_non_dominated_sort(F)
            # Calculate crowding distance for each front
            # We need a function that returns crowding distance for all points
            # Standard NSGA-II calculates it for the last front that needs truncation.
            # For selection, we can just use rank and then crowding.
            
            # Let's pre-calculate crowding for all fronts? 
            # It's expensive. 
            # Standard approach: 
            # Sort by rank. 
            # For the front that is partially selected, calculate crowding.
            
            # For simplicity in this implementation, let's just do random selection 
            # from the first few fronts? No, that's bad.
            
            # Let's implement a proper tournament selection.
            # We need a way to compare two individuals.
            # If rank[i] < rank[j], i is better.
            # If rank[i] == rank[j], compare crowding distance.
            
            # We will calculate crowding distances for all points in X.
            # This is O(N^2 * N_obj).
            
            cd = self.calculate_crowding_distances(F)
            
            for _ in range(num_offspring):
                # Tournament
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                
                if ranks[i1] < ranks[i2]:
                    p1 = X[i1]
                elif ranks[i2] < ranks[i1]:
                    p1 = X[i2]
                else:
                    if cd[i1] > cd[i2]:
                        p1 = X[i1]
                    else:
                        p1 = X[i2]
                        
                i3 = np.random.randint(0, pop_size)
                i4 = np.random.randint(0, pop_size)
                
                if ranks[i3] < ranks[i4]:
                    p2 = X[i3]
                elif ranks[i4] < ranks[i3]:
                    p2 = X[i4]
                else:
                    if cd[i3] > cd[i4]:
                        p2 = X[i3]
                    else:
                        p2 = X[i4]
                        
                # Crossover
                c1, c2 = self.sbx_crossover(p1, p2, lo, hi)
                # Mutation
                c1 = self.poly_mutation(c1, lo, hi)
                c2 = self.poly_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
                # Evaluate immediately to keep track of count?
                # Or batch evaluate?
                # Let's batch evaluate at the end of offspring generation.
            
            # Evaluate offspring
            # We generated 2 * num_offspring points?
            # No, the loop ran num_offspring times, generating 2 children each.
            # So total children = 2 * num_offspring.
            # We need to check if we have budget for all of them.
            # If not, we should only generate enough to fill the budget.
            
            # Let's adjust:
            # We want to generate exactly `self.budget - eval_count` children if possible, 
            # or just pop_size.
            
            # Let's simplify:
            # Generate pop_size children.
            # If budget is tight, generate fewer.
            
            # Re-implementing the offspring generation loop to be safer with budget.
            
            # Clear previous attempt
            offspring_X = []
            num_to_generate = min(2 * pop_size, self.budget - eval_count)
            # We generate pairs, so num_to_generate should be even?
            # If we generate 1 pair, that's 2 evals.
            
            # Let's just generate children one by one until budget is met or we have pop_size children.
            # Target offspring size: pop_size.
            
            while len(offspring_X) < pop_size and eval_count < self.budget:
                if time.time() > deadline:
                    break
                    
                # Tournament for p1
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                if ranks[i1] < ranks[i2]:
                    p1 = X[i1]
                elif ranks[i2] < ranks[i1]:
                    p1 = X[i2]
                else:
                    p1 = X[i1] if cd[i1] > cd[i2] else X[i2]
                    
                # Tournament for p2
                i3 = np.random.randint(0, pop_size)
                i4 = np.random.randint(0, pop_size)
                if ranks[i3] < ranks[i4]:
                    p2 = X[i3]
                elif ranks[i4] < ranks[i3]:
                    p2 = X[i4]
                else:
                    p2 = X[i3] if cd[i3] > cd[i4] else X[i4]
                    
                c1, c2 = self.sbx_crossover(p1, p2, lo, hi)
                c1 = self.poly_mutation(c1, lo, hi)
                c2 = self.poly_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                if len(offspring_X) < pop_size and eval_count < self.budget:
                    offspring_X.append(c2)
            
            if not offspring_X:
                break
                
            # Evaluate
            offspring_F = []
            for child in offspring_X:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                fc = problem(child)
                offspring_F.append(fc)
                eval_count += 1
            
            if not offspring_F:
                break
                
            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)
            
            # Combine
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Select
            new_X, new_F = self.select_population(X_combined, F_combined, pop_size)
            
            X = new_X
            F = new_F
            
        return X

    def select_population(self, X, F, pop_size):
        ranks, fronts = self.fast_non_dominated_sort(F)
        
        new_idx = []
        for front in fronts:
            if len(new_idx) + len(front) <= pop_size:
                new_idx.extend(front)
            else:
                # Need to select from this front
                needed = pop_size - len(new_idx)
                # Calculate crowding distance for points in this front
                # We need the indices of the front
                front_F = F[front]
                cd = self.calculate_crowding_distances(front_F)
                # Sort by cd descending
                sorted_indices = np.argsort(-cd)
                for i in range(needed):
                    new_idx.append(front[sorted_indices[i]])
                break
                
        return X[new_idx], F[new_idx]

    def fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_d = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        front = []
        for i in range(n):
            if n_d[i] == 0:
                front.append(i)
        if front:
            fronts.append(front)
            
        current_front = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
            current_front += 1
            
        # Create rank array
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
                
        return rank, fronts

    def calculate_crowding_distances(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        for k in range(self.n_obj):
            # Sort by objective k
            sorted_idx = np.argsort(F[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            range_val = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if range_val == 0:
                continue
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / range_val
                
        return cd

    def sbx_crossover(self, p1, p2, lo, hi):
        eta = 20
        child1 = np.zeros_like(p1)
        child2 = np.zeros_like(p2)
        for i in range(self.dim):
            u = np.random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (2 * (1 - u)) ** (-1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            child1[i] = np.clip(child1[i], lo[i], hi[i])
            child2[i] = np.clip(child2[i], lo[i], hi[i])
        return child1, child2

    def poly_mutation(self, x, lo, hi):
        eta = 20
        x_mut = np.zeros_like(x)
        for i in range(self.dim):
            delta = np.random.random()
            if delta <= 0.5:
                val = (2 * delta) ** (1 / (eta + 1)) - 1
            else:
                val = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
            x_mut[i] = x[i] + (hi[i] - lo[i]) * val
            x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
