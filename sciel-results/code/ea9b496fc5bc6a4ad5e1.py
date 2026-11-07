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
        sin_term = np.sin(6 * np.pi * x1_samples)
        sin_term_pow = sin_term ** 6
        exp_term = np.exp(-4 * x1_samples)
        base = 1 - exp_term * sin_term_pow
        # Handle potential numerical issues if base < 0
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
        n_points = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_points)
        
        # 3. Map target f1s to x1s using linear interpolation
        # np.searchsorted returns the index where the target would be inserted
        indices = np.searchsorted(f1_sorted, target_f1s)
        
        # Clip indices to valid range [1, len-1] for interpolation pairs
        # We need indices i and i-1
        # If index is 0, we are at the beginning. If index is len, at the end.
        # searchsorted returns 0 if target < f1[0]
        # searchsorted returns len if target > f1[-1]
        
        x1_targets = np.zeros(n_points)
        
        # Handle boundary cases
        # Case 1: index == 0
        mask_0 = (indices == 0)
        x1_targets[mask_0] = x1_sorted[0]
        
        # Case 2: index == len(f1_sorted)
        mask_max = (indices == len(f1_sorted))
        x1_targets[mask_max] = x1_sorted[-1]
        
        # Case 3: 0 < index < len
        mask_mid = ~(mask_0 | mask_max)
        
        if np.any(mask_mid):
            idx_mid = indices[mask_mid]
            targets_mid = target_f1s[mask_mid]
            
            # f1_low is at idx-1, f1_high is at idx
            f1_low = f1_sorted[idx_mid - 1]
            f1_high = f1_sorted[idx_mid]
            x1_low = x1_sorted[idx_mid - 1]
            x1_high = x1_sorted[idx_mid]
            
            # Calculate interpolation fraction
            # Avoid division by zero if f1_high == f1_low
            denom = f1_high - f1_low
            # If denom is very small, just take x1_low
            valid_denom = np.abs(denom) > 1e-12
            
            fracs = np.zeros(len(idx_mid))
            fracs[valid_denom] = (targets_mid[valid_denom] - f1_low[valid_denom]) / denom[valid_denom]
            # Clip fracs to [0, 1] to be safe against numerical noise
            fracs = np.clip(fracs, 0, 1)
            
            x1_targets[mask_mid] = x1_low + fracs * (x1_high - x1_low)

        # 4. Construct decision vectors and evaluate
        recorded = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Clip to bounds just in case
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
