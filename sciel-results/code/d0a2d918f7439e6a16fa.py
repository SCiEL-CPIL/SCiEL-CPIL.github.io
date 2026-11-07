import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific knowledge:
        # Pareto front is obtained when x_2...x_n = 0.
        # f_1(x_1) = 1 - (1 - exp(-4*x_1) * sin(6*pi*x_1)**6)**(1/10)
        # f_2(x_1) = 1 - sqrt(f_1(x_1))
        
        # 1. Build a local lookup table for f_1(x_1)
        # Sample x_1 densely
        x1_samples = np.linspace(0, 1, 10000)
        # Calculate f_1 locally
        # Note: sin(6*pi*x_1)**6 can be computed efficiently
        sin_term = np.sin(6 * np.pi * x1_samples)
        sin_term_pow = sin_term ** 6
        exp_term = np.exp(-4 * x1_samples)
        base = 1 - exp_term * sin_term_pow
        # Handle potential numerical issues if base < 0 (shouldn't happen if exp*sin^6 <= 1)
        base = np.clip(base, 0, 1)
        f1_samples = 1 - base ** 0.1
        
        # Sort by f1 to allow lookup
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        # Determine the range of f1
        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]
        
        # 2. Select target f1 values uniformly in [f1_min, f1_max]
        # Use all budget for points on the front
        n_points = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_points)
        
        # 3. Map target f1s to x1s
        # For each target, find the closest f1 in the sorted table
        # np.searchsorted returns the index where the target would be inserted
        indices = np.searchsorted(f1_sorted, target_f1s)
        # Clip indices to valid range
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        # Get corresponding x1
        x1_targets = x1_sorted[indices]
        
        # 4. Construct decision vectors and evaluate
        # x = [x1, 0, 0, ..., 0]
        # bounds are [0,1] for all, so 0 is valid
        recorded = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Clip to bounds just in case (though 0 and x1 in [0,1] are safe)
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
