"""Multi-objective optimizer for RE21 using Hypervolume-driven Multi-Objective Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a combination of sparse global sampling and weighted-sum local search.
    
    Strategy:
    1. Perform sparse random sampling to establish an initial population.
    2. Use a set of weight vectors (sum to 1) to define multiple sub-problems.
    3. For each weight vector, run L-BFGS-B local optimization starting from the 
       best point in the current population for that weight.
    4. Update the population with the results, maintaining a non-dominated set.
    5. Iterate until the budget or time limit is reached.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 320s limit (10s safety margin before 330s kill)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial random sampling
        n_init = min(20, self.budget // 4)
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Define weight vectors for multi-objective optimization
        n_weights = 10
        weights = []
        for i in range(n_weights):
            w1 = (i + 0.5) / n_weights
            w2 = 1.0 - w1
            weights.append(np.array([w1, w2]))
        
        # Keep track of best point for each weight
        best_for_weight = [None] * n_weights
        best_val_for_weight = [np.inf] * n_weight

        def add_point(x, f):
            nonlocal used
            used += 1
            F = np.vstack([F, [f]])
            X = np.vstack([X, [x]])
            
            # Update best for each weight
            for i, w in enumerate(weights):
                val = w[0] * f[0] + w[1] * f[1]
                if val < best_val_for_weight[i]:
                    best_val_for_weight[i] = val
                    best_for_weight[i] = x.copy()
        
        # Main loop: local search from promising points
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a starting point: prefer non-dominated points
            # Simple dominance check
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            if not np.any(~dominated):
                # All dominated, pick random
                start_idx = rng.integers(len(F))
            else:
                # Pick from non-dominated
                nondom_idx = np.where(~dominated)[0]
                start_idx = nondom_idx[rng.integers(len(nondom_idx))]
            
            start_x = X[start_idx].copy()
            start_f = F[start_idx].copy()
            
            # Try local optimization with a few different weights
            for _ in range(2):  # Try 2 local optimizations per iteration
                if time.time() > deadline:
                    break
                
                # Pick a random weight
                w_idx = rng.integers(n_weights)
                w = weights[w_idx]
                
                def objective(x):
                    return w[0] * problem(x)[0] + w[1] * problem(x)[1]
                
                try:
                    res = minimize(objective, start_x, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 50, 'ftol': 1e-8})
                    if res.success or res.fun < start_f[0] * w[0] + start_f[1] * w[1]:
                        new_x = res.x
                        new_f = problem(new_x)
                        add_point(new_x, new_f)
                except Exception:
                    pass
        
        return X
# EVOLVE-BLOCK-END
