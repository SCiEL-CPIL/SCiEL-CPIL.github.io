import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # 1. Initial Sampling: Latin Hypercube for better coverage
        n_init = min(40, self.budget)
        X_init = np.empty((n_init, dim))
        for i in range(dim):
            X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
        idx = np.random.permutation(n_init)
        X_init = X_init[idx]
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Track all evaluated points for potential final return (though return is ignored, 
        # the contract says "Every returned objective vector is recorded", implying we just 
        # need to call problem. The return value of __call__ is ignored. 
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This usually means the evaluator intercepts `problem` calls. 
        # So we just need to make sure we call `problem` correctly.
        # The return value of `__call__` is ignored, so we can return anything.
        
        # 2. Setup Population for Evolution
        pop_size = 20
        if self.budget < pop_size + 1:
            # If budget is very small, just return what we have
            return X_init
            
        # Initialize population from initial samples
        # Use non-dominated points if available, otherwise random
        nd_idx = self._get_non_dominated(F_init)
        
        X_pop = np.empty((pop_size, dim))
        F_pop = np.empty((pop_size, n_obj))
        
        if len(nd_idx) >= pop_size:
            X_pop = X_init[nd_idx[:pop_size]].copy()
            F_pop = F_init[nd_idx[:pop_size]].copy()
        else:
            X_pop[:len(nd_idx)] = X_init[nd_idx]
            F_pop[:len(nd_idx)] = F_init[nd_idx]
            remaining = pop_size - len(nd_idx)
            all_idx = np.arange(len(X_init))
            nd_set = set(nd_idx.tolist())
            available = [i for i in all_idx if i not in nd_set]
            if len(available) >= remaining:
                fill_idx = np.random.choice(available, remaining, replace=False)
            else:
                fill_idx = np.random.choice(len(X_init), remaining, replace=True)
            X_pop[len(nd_idx):] = X_init[fill_idx]
            F_pop[len(nd_idx):] = F_init[fill_idx]
            
        # 3. Evolutionary Loop (DE-like with NSGA-II selection)
        F_mut = 0.8
        CR = 0.5
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate one offspring at a time to save budget and allow frequent checks
            i = np.random.randint(pop_size)
            
            # Select three distinct random indices r1, r2, r3 != i
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                # If pop size is small, just pick with replacement
                r1, r2, r3 = np.random.choice(pop_size, size=3, replace=True)
            else:
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
            
            # Mutation (DE/rand/1)
            v = X_pop[r1] + F_mut * (X_pop[r2] - X_pop[r3])
            
            # Crossover (Binomial)
            j_rand = np.random.randint(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X_pop[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection
            # Check if trial dominates i
            if np.all(f_trial <= F_pop[i]) and np.any(f_trial < F_pop[i]):
                X_pop[i] = trial
                F_pop[i] = f_trial
            elif not (np.all(F_pop[i] <= f_trial) and np.any(F_pop[i] < f_trial)):
                # Non-dominated: use crowding distance to decide
                # Calculate crowding distance for i and trial in the current population
                # For simplicity, we'll use a heuristic: replace if trial has better "spread"
                # or just use random with slight bias toward exploration
                
                # Compute simple crowding: sum of distances to neighbors in objective space
                # We only compare i and trial against the rest of the population
                # Let's implement a quick crowding check for just these two points relative to the pop
                # Actually, full crowding is expensive. Let's use a simpler diversity metric.
                # If they are non-dominated, we replace with probability 0.5 to maintain diversity.
                # But we can be smarter: check if trial is closer to the "ideal" point (min of mins)
                # or further from the current Pareto front.
                
                # Simple heuristic: replace if it's further from the mean of the population in objective space
                # This encourages exploration.
                mean_F = np.mean(F_pop, axis=0)
                dist_i = np.linalg.norm(F_pop[i] - mean_F)
                dist_trial = np.linalg.norm(f_trial - mean_F)
                
                # Replace if trial is more diverse (further from mean)
                if dist_trial > dist_i:
                    X_pop[i] = trial
                    F_pop[i] = f_trial
                elif np.random.random() < 0.3:
                    # Some randomness to avoid local optima
                    X_pop[i] = trial
                    F_pop[i] = f_trial

        return X_pop

    def _get_non_dominated(self, F):
        """
        Returns indices of non-dominated points in F.
        F is (N, n_obj). Minimization assumed.
        """
        N = len(F)
        if N == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(N, dtype=bool)
        
        for i in range(N):
            if is_dominated[i]:
                continue
            for j in range(N):
                if i == j or is_dominated[j]:
                    continue
                # Check if i dominates j
                # i dominates j if i <= j and i < j in at least one dimension
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
        
        return np.where(~is_dominated)[0]
