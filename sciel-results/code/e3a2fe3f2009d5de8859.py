"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Key insight: For DTLZ7 with m=3 objectives and n=22 variables:
    - The first m-1 = 2 variables (x[0], x[1]) directly set f1, f2
    - x[2] is part of the "inactive" set that influences g
    - The last n - (m-1) = 20 variables influence g
    
    Actually, let me reconsider DTLZ7 structure:
    - f1 = x[0]
    - f2 = x[1]  
    - f3 = g(x_in) * h(x[2], g, ...)
    where x_in = x[m-1:] = x[2:] (the last 20 variables)
    
    Wait, standard DTLZ7:
    - m objectives, n variables
    - f_i = x_{i-1} for i = 1, ..., m-1
    - f_m = g(x_in) * h(f_{m-1}, g, x_{m-1})
    where x_in = x_{n-k+1}, ..., x_n (last k variables, k = n - m + 1)
    
    For n=22, m=3: k = 22 - 3 + 1 = 20
    So x_in = x[2], x[3], ..., x[21] (indices 2 through 21, that's 20 variables)
    
    g(x_in) = 12 * n * x_in / (1 + sum(x_in))
    h(y, g, x) = 1 - sqrt(y/g) * (1 - x/(g+1) * (1 - sin(3*pi*x/(g+1))))
    
    The Pareto front is achieved when g = 1 (minimum of g when x_in are at their minimum).
    Actually, g minimum: when all x_in = 0, g = 0? Let me check.
    
    g(x_in) = 12 * n * x_in / (1 + sum(x_in))
    If x_in are all 0: g = 0
    But h involves division by g, so we need g > 0.
    
    The Pareto front for DTLZ7 is complex. The key is that to get good hypervolume,
    we need points close to the Pareto front.
    
    Strategy: 
    1. Sample x[0] and x[1] uniformly in [0,1] to cover f1 and f2
    2. For the remaining variables, use a strategy that minimizes g
    3. Use a grid or quasi-random sampling to efficiently explore the front
    
    Since we have budget=200 evaluations, we can afford a structured search.
    
    Better approach: Use NSGA-II or a similar MOEA, but with the knowledge that
    the Pareto front structure is known.
    
    Actually, the most effective approach for DTLZ7 with known structure:
    - Fix the "position" variables (x[0], x[1]) to spread out f1, f2
    - Optimize the remaining variables to minimize g (or get close to the PF)
    
    Let me implement a hybrid:
    1. Generate diverse points by varying x[0], x[1]
    2. For each, set the other variables to minimize g
    3. Use the remaining budget to refine
    
    For g minimization: g = 12*n*sum(x_in)/(1+sum(x_in)) where x_in are the last 20 vars
    To minimize g, we want sum(x_in) small. Setting all x_in = 0 gives g=0,
    but then h has issues. Let's set x_in to small values.
    
    Actually, the Pareto front of DTLZ7 is non-dominated when g is at its minimum
    achievable value. The minimum of g over x_in in [0,1]^20 is 0 (at x_in=0).
    
    Let me just implement a robust multi-objective optimizer that works well:
    Use a population-based approach with non-dominated sorting and crowding.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use the seed from np.random if set, otherwise use default
        # The evaluator calls np.random.seed(seed) before creating Optimizer
        # So we should use np.random for reproducibility with the seed
        
        # Population size
        pop_size = min(50, max(20, budget // 4))
        
        # Initialize population with diverse points
        # For DTLZ7, focus on the key dimensions
        X = np.zeros((pop_size, dim))
        # x[0] and x[1] control f1 and f2 directly
        # Spread them out
        X[:, 0] = np.linspace(0.01, 0.99, pop_size)
        X[:, 1] = np.linspace(0.01, 0.99, pop_size)
        # Other dimensions: start near 0 to minimize g
        for d in range(2, dim):
            X[:, d] = np.random.uniform(0, 0.1, pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_F = [F.copy()]
        
        def non_dominated_sort(F):
            """Return list of non-dominated fronts (list of index lists)."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_by = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_by[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_by[j].append(i)
                        domination_count[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distances for the given indices."""
            n = len(indices)
            if n == 0:
                return np.array([])
            if n == 1:
                return np.array([np.inf])
            
            distances = np.zeros(n)
            F_subset = F[indices]
            
            for k in range(n_obj):
                sorted_idx = np.argsort(F_subset[:, k])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                if F_subset[sorted_idx[-1], k] - F_subset[sorted_idx[0], k] > 1e-15:
                    for i in range(1, n - 1):
                        distances[sorted_idx[i]] += (
                            F_subset[sorted_idx[i + 1], k] - F_subset[sorted_idx[i - 1], k]
                        ) / (F_subset[sorted_idx[-1], k] - F_subset[sorted_idx[0], k])
            
            return distances
        
        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            
            # Select parents from the first front
            if len(fronts[0]) > 0:
                parent_indices = fronts[0]
            else:
                parent_indices = list(range(pop_size))
            
            # Create offspring
            n_offspring = min(pop_size // 2, budget - used)
            if n_offspring <= 0:
                break
            
            offspring_X = np.zeros((n_offspring, dim))
            
            for i in range(n_offspring):
                # Pick a parent
                p_idx = parent_indices[np.random.randint(len(parent_indices))]
                parent = X[p_idx]
                
                # Mutation: perturb all dimensions
                child = parent.copy()
                
                # For DTLZ7, focus mutations on key dimensions
                # x[0], x[1] get larger steps, others smaller
                for d in range(dim):
                    if d < 2:
                        step = 0.1 * np.random.randn()
                    else:
                        step = 0.05 * np.random.randn()
                    child[d] += step
                
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            all_F.append(offspring_F.copy())
            
            # Combine population and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Select next generation
            fronts = non_dominated_sort(combined_F)
            
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            filled = 0
            
            for front in fronts:
                if filled + len(front) <= pop_size:
                    for idx in front:
                        new_X[filled] = combined_X[idx]
                        new_F[filled] = combined_F[idx]
                        filled += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - filled
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X[filled] = combined_X[idx]
                        new_F[filled] = combined_F[idx]
                        filled += 1
                    break
            
            X = new_X
            F = new_F
        
        # Return (the return value is ignored, but we need to return something)
        return X
# EVOLVE-BLOCK-END
