import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 40
        # We have 200 budget. 
        # Init: 40
        # Gen 1: 40 offspring -> 80 combined -> select 40
        # Gen 2: 40 offspring -> 80 combined -> select 40
        # Gen 3: 40 offspring -> 80 combined -> select 40
        # Gen 4: 40 offspring -> 80 combined -> select 40
        # Total: 40 + 4*40 = 200.
        
        # Local search will steal budget from generations if we add it.
        # Let's do 3 generations of EA (120 evals) + 80 evals of Local Search?
        # Or just run EA and use local search on the final front?
        # With 200 budget, 3 generations of NSGA-II is too few to converge.
        
        # Alternative: Smaller population, more generations?
        # Pop size 20.
        # Init: 20
        # 9 generations of 20 offspring = 180. Total 200.
        
        pop_size = 20
        n_generations = (self.budget - pop_size) // pop_size
        
        # Initialize with Latin Hypercube
        X = self._latin_hypercube(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        
        used = pop_size
        
        for gen in range(n_generations):
            # Check time
            if time.time() - self.start_time > 280:
                break
                
            # Selection
            parents_idx = self._nsga2_select(X, F, pop_size)
            parents = X[parents_idx]
            
            # Crossover and Mutation
            children = []
            for i in range(0, pop_size - 1, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                # SBX
                c1, c2 = self._sbx(p1, p2, lo, hi)
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                children.append(c1)
                children.append(c2)
            
            X_child = np.array(children)
            F_child = np.array([problem(x) for x in X_child])
            used += len(X_child)
            
            # Combine
            X_comb = np.vstack([X, X_child])
            F_comb = np.vstack([F, F_child])
            
            # Non-dominated sorting and crowding
            X, F = self._nsga2_env_selection(X_comb, F_comb, pop_size)
            
        # Local Search on the final non-dominated set
        # Use remaining budget
        remaining = self.budget - used
        if remaining > 0:
            X, F = self._local_search(X, F, problem, lo, hi, remaining)
            
        return X

    def _latin_hypercube(self, n, dim, lo, hi):
        # Simple LHS
        X = np.zeros((n, dim))
        for j in range(dim):
            # Divide [0,1] into n intervals, pick one random point in each
            indices = np.random.permutation(n)
            for i in range(n):
                # i-th interval for the i-th sample? No, standard LHS:
                # For each dimension, assign a unique interval to each sample.
                pass
        
        # Easier LHS:
        X = np.zeros((n, dim))
        for j in range(dim):
            u = np.random.rand(n)
            # Sort u to ensure one per bin?
            # Actually, standard LHS:
            # For each dimension j, generate n uniform random numbers, sort them.
            # The i-th smallest number goes to the i-th row? No.
            # The i-th row gets the i-th number from the sorted list?
            # Yes, this ensures one point in each bin [i/n, (i+1)/n].
            u_sorted = np.sort(u)
            X[:, j] = u_sorted
        
        # Map to [lo, hi]
        X = lo + X * (hi - lo)
        return X

    def _sbx(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        # eta = 20
        eta = 20
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                # Calculate bounds for child
                # Standard SBX formulas
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Clip
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        # eta = 20
        eta = 20
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def _nsga2_select(self, X, F, k):
        # Tournament selection
        ranks, distances = self._non_dominated_sort(F)
        parents = []
        while len(parents) < k:
            # Tournament of 2
            idx1 = np.random.randint(0, len(X))
            idx2 = np.random.randint(0, len(X))
            if ranks[idx1] < ranks[idx2]:
                parents.append(idx1)
            elif ranks[idx2] < ranks[idx1]:
                parents.append(idx2)
            else:
                if distances[idx1] > distances[idx2]:
                    parents.append(idx1)
                else:
                    parents.append(idx2)
        return parents

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        distances = np.zeros(n)
        
        # Calculate domination
        dominated_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    dominated_count[i] += 1
        
        # Non-dominated set
        front = [i for i in range(n) if dominated_count[i] == 0]
        ranks[front] = 0
        
        current_rank = 0
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            front = next_front
            current_rank += 1
        
        # Crowding distance
        # For each rank, calculate distance
        unique_ranks = np.unique(ranks)
        for r in unique_ranks:
            idxs = np.where(ranks == r)[0]
            if len(idxs) <= 2:
                distances[idxs] = np.inf
                continue
            
            # Normalize objectives for this rank? No, use original values.
            # Sort by each objective
            for m in range(self.n_obj):
                sorted_idxs = idxs[np.argsort(F[idxs, m])]
                distances[sorted_idxs[0]] = np.inf
                distances[sorted_idxs[-1]] = np.inf
                for k in range(1, len(sorted_idxs) - 1):
                    i = sorted_idxs[k]
                    i_prev = sorted_idxs[k-1]
                    i_next = sorted_idxs[k+1]
                    dist = (F[i_next, m] - F[i_prev, m])
                    if dist > 0:
                        distances[i] += dist
                    else:
                        distances[i] += 1e-10
                        
        return ranks, distances

    def _nsga2_env_selection(self, X, F, k):
        ranks, distances = self._non_dominated_sort(F)
        
        # Sort by rank, then by distance (descending)
        # We want the best k
        # Create a list of indices sorted by (rank, -distance)
        sorted_indices = np.argsort(ranks - distances) # This is not quite right because ranks are small integers and distances are large.
        
        # Better: Stable sort by rank, then by distance descending
        # Use lexsort
        # We want min rank, max distance
        # So sort by rank (ascending), then by -distance (ascending)
        sorted_indices = np.lexsort((-distances, ranks))
        
        selected_indices = sorted_indices[:k]
        return X[selected_indices], F[selected_indices]

    def _local_search(self, X, F, problem, lo, hi, budget):
        # Simple local search on non-dominated points
        # We assume X and F are the current population (non-dominated or best k)
        # We will iterate through points and try to improve them
        # Budget is limited
        
        n = len(X)
        if n == 0:
            return X, F
        
        # Identify non-dominated points in the current set
        # Since _nsga2_env_selection returns the best k, they are likely non-dominated or close
        # Let's just try to improve all of them
        
        # Step size
        step_size = 0.05 * (hi - lo)
        
        for i in range(n):
            if budget <= 0:
                break
                
            x = X[i].copy()
            f = F[i].copy()
            
            # Try coordinate-wise perturbations
            improved = False
            for dim in range(self.dim):
                if budget <= 0:
                    break
                
                # Try + and -
                for sign in [1, -1]:
                    if budget <= 0:
                        break
                    
                    x_new = x.copy()
                    x_new[dim] += sign * step_size[dim]
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    budget -= 1
                    
                    # Check if f_new is better than f (dominates or is non-dominated but better in some way?)
                    # For local search in MO, we usually accept if it is not dominated by any other point in the set
                    # And ideally improves the hypervolume
                    # Simple rule: Accept if f_new is not dominated by any other point in F
                    # And f_new dominates f? No, just non-dominated.
                    
                    # Check if f_new is dominated by any existing point in F (excluding itself)
                    dominated = False
                    for j in range(n):
                        if j == i:
                            continue
                        if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                            dominated = True
                            break
                    
                    if not dominated:
                        # Replace if it is better than current f in at least one objective and not worse in others?
                        # Or just if it is non-dominated?
                        # Let's say we replace if it dominates the current point OR if it is non-dominated and has a lower sum?
                        # To be safe, replace if it dominates the current point.
                        if np.all(f_new <= f) and np.any(f_new < f):
                            X[i] = x_new
                            F[i] = f_new
                            improved = True
                            break # Move to next point
                        else:
                            # If not dominating, maybe keep the best of the two?
                            # For simplicity, only accept if it dominates.
                            pass
                            
        return X, F
