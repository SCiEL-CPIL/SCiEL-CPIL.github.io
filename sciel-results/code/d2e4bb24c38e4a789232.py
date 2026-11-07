import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        # WFG9: f1 = x1, f2 and f3 depend on x1 and other variables
        # The Pareto front is achieved when the "position" variables (x2-x12) 
        # are set to minimize the shape functions
        # For WFG9, the ideal is to have x2-x12 at specific values that minimize the objectives

        # Strategy: 
        # 1. Create initial diverse population covering the x1 dimension
        # 2. For each x1 value, try to find good values for other variables
        # 3. Use local search to refine

        N_init = 30
        N_total = 200

        # Generate initial points with diverse x1 values
        # WFG9: x1 controls the trade-off, other variables affect the shape
        X = np.zeros((N_init, self.dim))
        # Spread x1 across its range
        x1_values = np.linspace(lo[0], hi[0], N_init)
        for i in range(N_init):
            X[i, 0] = x1_values[i]
            # For other variables, use mid-range initially, then refine
            X[i, 1:] = mid[1:] + np.random.randn(self.dim - 1) * span[1:] * 0.1

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N_init

        # Helper: fast non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while k < len(fronts) and fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = F.shape[1]
            
            for k in range(m):
                sorted_idx = np.argsort(F[front_indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[front_indices, k].max() - F[front_indices, k].min()
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F[front_indices[sorted_idx[i + 1]], k] - 
                                           F[front_indices[sorted_idx[i - 1]], k]) / f_range
            return cd

        # Main loop: use remaining budget for local search
        # Keep a population of non-dominated solutions
        pop_size = 40
        
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            # Get non-dominated fronts
            fronts = non_dominated_sort(F)
            front0 = fronts[0]
            
            # Select parent from first front using crowding distance
            if len(front0) == 1:
                parent_idx = front0[0]
            else:
                cd = crowding_distance(F, front0)
                parent_idx = front0[np.argmax(cd)]
            
            x_parent = X[parent_idx]
            
            # Adaptive step size based on how many evaluations we've used
            progress = used / self.budget
            base_step = 0.15 * span
            step_size = base_step * (1.0 - 0.5 * progress)
            
            # Generate multiple offspring with different perturbations
            n_offspring = min(5, self.budget - used)
            
            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > limit_time:
                    break
                
                # Mix of Gaussian and uniform perturbations
                if np.random.rand() < 0.5:
                    x_child = x_parent + np.random.normal(0, step_size)
                else:
                    x_child = x_parent + np.random.uniform(-step_size, step_size, self.dim)
                x_child = np.clip(x_child, lo, hi)
                
                f_child = problem(x_child)
                used += 1
                
                # Add to population
                X = np.vstack([X, x_child])
                F = np.vstack([F, f_child])
            
            # Prune population: keep non-dominated, then most diverse
            if len(X) > pop_size:
                fronts = non_dominated_sort(F)
                keep_indices = []
                for front in fronts:
                    if len(keep_indices) + len(front) <= pop_size:
                        keep_indices.extend(front)
                    else:
                        # Need to select from this front
                        remaining = pop_size - len(keep_indices)
                        cd = crowding_distance(F, front)
                        top_indices = np.argsort(cd)[-remaining:]
                        keep_indices.extend([front[i] for i in top_indices])
                        break
                
                if len(keep_indices) > 0:
                    X = X[keep_indices]
                    F = F[keep_indices]
        
        return X
