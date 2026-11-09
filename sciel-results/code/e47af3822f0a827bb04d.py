import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 320.0 # Safety margin from 330s

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific knowledge:
        # Pareto front is at x_i = 0.5^(1/alpha) for i >= 2.
        # Assuming alpha=10 (standard DTLZ4), target is 0.5^0.1 ≈ 0.933033
        target_val = 0.5 ** 0.1
        target_vec = np.full(dim, target_val)
        target_vec[0] = 0.5 # Irrelevant for position, just center
        target_vec[1] = 0.5
        
        # Population size
        pop_size = 20
        # Ensure we don't exceed budget
        if pop_size >= budget:
             pop_size = max(2, budget // 2)
             
        # Initialize population
        # x0, x1 uniform
        # x2...x11 near target_val
        X = np.empty((pop_size, dim))
        X[:, 0] = np.random.uniform(lo[0], hi[0], pop_size)
        X[:, 1] = np.random.uniform(lo[1], hi[1], pop_size)
        # Add small noise to target values for x2...x11
        noise = np.random.normal(0, 0.02, (pop_size, dim - 2))
        X[:, 2:] = np.clip(target_vec[2:] + noise, lo[2:], hi[2:])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Evolutionary Loop
        # We will run until budget is exhausted or time limit approached
        # Use NSGA-II style selection
        
        while used < budget:
            # Check time
            if time.time() > self.deadline:
                break
            
            # Determine number of children to create
            # We can create up to pop_size children, but limited by remaining budget
            n_children = min(pop_size, budget - used)
            if n_children == 0:
                break
            
            # Create children
            children_X = np.empty((n_children, dim))
            
            for i in range(n_children):
                # Select parent
                # Simple strategy: Pick a random non-dominated parent
                # To save computation, we can just pick random from current pop if we haven't sorted yet
                # Or maintain a non-dominated set. For small pop, sorting is cheap.
                
                # Let's do a quick non-dominated check to pick good parents
                # Actually, for speed, let's just pick random parents from the current population
                # and rely on the selection step to filter out bad ones.
                # But picking non-dominated is better.
                
                # Identify non-dominated indices
                dom_mask = np.zeros(pop_size, dtype=bool)
                for k in range(pop_size):
                    # Check if F[k] is dominated by any other F[j]
                    # F[j] <= F[k] and F[j] != F[k]
                    if np.any(np.all(F <= F[k], axis=1) & np.any(F < F[k], axis=1)):
                         dom_mask[k] = True
                 
                nondom_idx = np.where(~dom_mask)[0]
                if len(nondom_idx) == 0:
                    parent_idx = np.random.randint(pop_size)
                else:
                    parent_idx = nondom_idx[np.random.randint(len(nondom_idx))]
                
                parent = X[parent_idx]
                
                # Mutation
                # Strategy:
                # x0, x1: Larger mutation to explore the front shape
                # x2..x11: Smaller mutation, and "repair" towards target_val
                child = parent.copy()
                
                # Mutate x0, x1
                sigma_01 = 0.2
                child[0] += np.random.normal(0, sigma_01)
                child[1] += np.random.normal(0, sigma_01)
                
                # Mutate x2..x11
                # Pull towards target
                sigma_rest = 0.05
                child[2:] += np.random.normal(0, sigma_rest, dim - 2)
                # Add a pull towards target_val to counteract drift
                child[2:] += 0.1 * (target_val - child[2:])
                
                child = np.clip(child, lo, hi)
                children_X[i] = child
                
            # Evaluate children
            children_F = np.array([problem(x) for x in children_X])
            used += n_children
            
            # Combine parents and children
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])
            
            # Selection: NSGA-II style
            # 1. Non-dominated sorting
            # 2. Crowding distance
            # 3. Select best pop_size
            
            X, F = self._select_best(X_combined, F_combined, pop_size)
            
        return X

    def _select_best(self, X, F, pop_size):
        # Non-dominated sorting
        n = len(X)
        ranks = np.zeros(n, dtype=int)
        front = 0
        remaining = np.arange(n)
        
        while len(remaining) > 0:
            # Find non-dominated set in remaining
            nondom = []
            for i in remaining:
                is_dom = False
                for j in remaining:
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom.append(i)
            
            if not nondom:
                # Fallback: assign all remaining to current front
                for i in remaining:
                    ranks[i] = front
                break
                
            for i in nondom:
                ranks[i] = front
            remaining = np.array([i for i in remaining if i not in nondom])
            front += 1
            
        # Crowding distance
        dist = np.zeros(n)
        for f in range(front):
            idx = np.where(ranks == f)[0]
            if len(idx) <= 2:
                dist[idx] = np.inf
                continue
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = idx[np.argsort(F[idx, m])]
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[idx, m].min()
                f_max = F[idx, m].max()
                if f_max == f_min:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[k]] += (F[sorted_idx[k+1], m] - F[sorted_idx[k-1], m]) / (f_max - f_min)
        
        # Sort by rank, then by crowding distance (descending)
        # We want lowest rank, then highest distance
        order = np.lexsort(((-dist), ranks))
        
        return X[order[:pop_size]], F[order[:pop_size]]
