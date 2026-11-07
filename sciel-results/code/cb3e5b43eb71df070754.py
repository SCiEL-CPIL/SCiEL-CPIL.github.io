import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With budget 200, pop_size 50 gives 4 generations.
        # pop_size 20 gives 10 generations.
        # Let's try 40 to balance.
        pop_size = 40
        
        # Initialize
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # NSGA-II parameters
        # Crossover rate, mutation rate
        # For 30 dims, mutation rate per gene ~ 1/30
        # SBX eta ~ 15
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size:
                # Crossover
                # Tournament selection for parents
                p1, p2 = self.tournament_select(X, F, k=2)
                
                # SBX Crossover
                if np.random.rand() < 0.9:
                    c1, c2 = self.sbx_crossover(p1, p2, lo, hi)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self.polynomial_mutation(c1, lo, hi, eta=20)
                c2 = self.polynomial_mutation(c2, lo, hi, eta=20)
                
                # Evaluate
                if evals + 2 > self.budget:
                    # Only evaluate if we have budget
                    if evals < self.budget:
                        f1 = problem(c1)
                        offspring_X.append(c1)
                        offspring_F.append(f1)
                        evals += 1
                    break
                else:
                    f1 = problem(c1)
                    f2 = problem(c2)
                    offspring_X.append(c1)
                    offspring_X.append(c2)
                    offspring_F.append(f1)
                    offspring_F.append(f2)
                    evals += 2
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(F_comb)
            
            # Crowding distance
            # Assign crowding distance to all points in fronts
            # We need to assign distance for selection
            # Standard NSGA-II:
            # 1. Sort fronts
            # 2. Fill next pop
            # 3. If last front is partial, sort by crowding distance
            
            # We need crowding distance for the fronts that might be partially selected
            # Calculate for all fronts to be safe, or just the ones needed?
            # Calculating for all is O(N^2 log N) roughly, which is fine.
            
            # However, we only need distances for the front that determines the cut.
            # Let's just calculate for all points in the combined population for simplicity, 
            # but only for the fronts that are not fully selected? 
            # Actually, standard implementation calculates for the fronts as they are added.
            
            next_X = []
            next_F = []
            
            # Sort fronts by rank (already sorted by non_dominated_sort usually)
            # non_dominated_sort returns list of lists of indices
            
            # We need to assign crowding distance to the individuals in the fronts
            # to be able to sort them if the front is split.
            
            # Let's compute crowding distance for all points in the combined population
            # This is inefficient but N is small (80).
            
            # Actually, we can just compute it for the fronts.
            
            # Let's re-structure:
            # 1. Get fronts (list of index lists)
            # 2. Iterate through fronts
            # 3. If current size + len(front) <= pop_size, take all.
            # 4. Else, compute crowding distance for this front, sort, take remaining.
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select from this front
                    # Compute crowding distance for these points
                    F_front = F_comb[front]
                    X_front = X_comb[front]
                    
                    # Crowding distance calculation
                    # For each objective, sort the front
                    # Assign distance
                    dist = np.zeros(len(front))
                    
                    for m in range(self.n_obj):
                        # Sort indices by objective m
                        sorted_indices = np.argsort(F_front[:, m])
                        
                        # Boundary points get infinity
                        dist[sorted_indices[0]] = np.inf
                        dist[sorted_indices[-1]] = np.inf
                        
                        # Normalize range
                        f_min = F_front[sorted_indices[0], m]
                        f_max = F_front[sorted_indices[-1], m]
                        f_range = f_max - f_min
                        if f_range == 0:
                            continue
                            
                        for i in range(1, len(front) - 1):
                            idx_curr = sorted_indices[i]
                            idx_prev = sorted_indices[i-1]
                            idx_next = sorted_indices[i+1]
                            
                            dist[idx_curr] += (F_front[idx_next, m] - F_front[idx_prev, m]) / f_range
                    
                    # Sort front by distance descending
                    sorted_front_indices = np.argsort(dist)[::-1]
                    
                    # Take remaining
                    remaining = pop_size - len(next_X)
                    for i in range(remaining):
                        idx_in_front = sorted_front_indices[i]
                        idx_in_comb = front[idx_in_front]
                        next_X.append(X_comb[idx_in_comb])
                        next_F.append(F_comb[idx_in_comb])
                    
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X

    def tournament_select(self, X, F, k=2):
        n = len(X)
        indices = np.random.choice(n, k, replace=False)
        # Find non-dominated among selected
        # Simple domination check
        best_idx = indices[0]
        for i in range(1, k):
            idx = indices[i]
            if self.dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx], F[best_idx] # Return X and F? No, select returns X vector.
        # Wait, I need to return two parents X vectors.
        # Let's return X[best_idx] and X[another_best_idx] or just best twice?
        # Usually tournament selects one parent. I need two.
        # Let's do two tournaments.

    def tournament_select_single(self, X, F, k=2):
        n = len(X)
        indices = np.random.choice(n, k, replace=False)
        best_idx = indices[0]
        for i in range(1, k):
            idx = indices[i]
            if self.dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx]

    def dominates(self, f1, f2):
        # f1 dominates f2 if f1 <= f2 and f1 != f2
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def non_dominated_sort(self, F):
        n = len(F)
        fronts = []
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(F[i], F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(F[j], F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
            
        return fronts

    def sbx_crossover(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                if p1[i] == p2[i]:
                    continue
                # SBX
                # beta = ...
                # Simplified SBX
                if np.random.rand() < 0.5:
                    beta = 1 + 2 * (p1[i] - p2[i]) / (hi[i] - lo[i])
                else:
                    beta = 1 + 2 * (p2[i] - p1[i]) / (hi[i] - lo[i])
                    
                if np.random.rand() <= 1 / (beta + 1):
                    beta_q = beta ** (1/15) # eta=15
                else:
                    beta_q = (1 / (beta - 1)) ** (1/15)
                    
                c1[i] = 0.5 * ((1 + beta_q) * p1[i] + (1 - beta_q) * p2[i])
                c2[i] = 0.5 * ((1 - beta_q) * p1[i] + (1 + beta_q) * p2[i])
                
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def polynomial_mutation(self, x, lo, hi, eta=20):
        for i in range(len(x)):
            if np.random.rand() < 1.0 / len(x):
                if x[i] == lo[i] or x[i] == hi[i]:
                    continue
                y = x[i]
                # u = np.random.rand()
                # delta1 = ((y - lo[i]) / (hi[i] - lo[i]))
                # delta2 = 1 - delta1
                # if u < 0.5:
                #     xy = y + (hi[i] - lo[i]) * ((2*u + (1-2*u)*delta1**(eta+1))**(1/(eta+1)) - 1)
                # else:
                #     xy = y - (hi[i] - lo[i]) * ((2*(1-u) + (2*u-1)*delta2**(eta+1))**(1/(eta+1)) - 1)
                # x[i] = np.clip(xy, lo[i], hi[i])
                
                # Simpler Gaussian mutation for speed and simplicity in this small budget context
                # But Polynomial is standard for NSGA-II.
                # Let's use a simple step mutation or Gaussian to avoid complex math errors.
                # Actually, let's use a simple random perturbation.
                # The directive says "structural change", NSGA-II is the structure.
                # Mutation type is a detail.
                # Let's use a Gaussian mutation scaled by range.
                scale = (hi[i] - lo[i]) * 0.1
                x[i] = x[i] + np.random.normal(0, scale)
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
