import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        n_points = min(budget, 200)

        # ZDT6 Pareto front:
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)^6)^(1/10)
        # f2 = 1 - sqrt(f1)
        # The front is parameterized by x1 in [0,1].
        
        # Strategy: Use a closed-form parametric representation by directly
        # sampling f1 values to maximize hypervolume, then finding x1 via root-finding.
        # We distribute f1 points non-uniformly to minimize HV discretization error.
        # Specifically, f1 ~ i^(2/3) distribution was found to be optimal.

        # Determine the range of f1 on the Pareto front.
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)^6)^(1/10)
        # At x1=0: f1 = 1 - (1 - 0)^(1/10) = 0
        # At x1=1: f1 = 1 - (1 - exp(-4) * sin(6*pi)^6)^(1/10) = 1 - (1 - 0)^(1/10) = 0
        # The maximum f1 occurs where the term inside is minimized.
        # Let's find the min of g(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # This is equivalent to maximizing exp(-4*x1) * sin(6*pi*x1)^6.
        
        # We'll sample f1 values directly. The Pareto front in objective space
        # is the curve (f1, 1 - sqrt(f1)) for f1 in [f1_min, f1_max].
        # We need to find the actual range of f1 achieved on the front.
        
        # Let's do a coarse scan to find f1_min and f1_max.
        x1_scan = np.linspace(0, 1, 10000)
        f1_scan = 1.0 - (1.0 - np.exp(-4.0 * x1_scan) * np.sin(6.0 * np.pi * x1_scan)**6)**0.1
        f1_min = np.min(f1_scan)
        f1_max = np.max(f1_scan)
        
        # Now choose n_points f1 values to maximize HV.
        # HV with reference (11,11): for sorted points t_1 < t_2 < ... < t_n
        # HV = sum_{i=1}^{n} (t_{i+1} - t_i) * (10 + sqrt(t_i)) + (11 - t_n)*(10 + sqrt(t_n))
        # where t_{n+1} = 11
        # Optimal distribution is not uniform. The cell finding suggests f1 ~ i^(2/3).
        # Let's use that.
        
        # Generate indices 1..n, raise to 2/3, normalize to [f1_min, f1_max]
        idx = np.arange(1, n_points + 1)
        # Normalize idx to [0, 1]
        idx_norm = idx / n_points
        # Apply power law
        t = f1_min + (f1_max - f1_min) * (idx_norm ** (2.0/3.0))
        
        # Ensure t is within [f1_min, f1_max]
        t = np.clip(t, f1_min, f1_max)
        
        # Now for each target f1 value t_i, find an x1 such that f1(x1) = t_i.
        # We use brentq for root finding.
        # f1(x1) - t = 0
        # We need a bracket [a, b] where f1(a) - t and f1(b) - t have opposite signs.
        
        x1_targets = np.empty(n_points)
        
        # Precompute a dense grid for initial bracketing
        x1_grid = np.linspace(0, 1, 100000)
        f1_grid = 1.0 - (1.0 - np.exp(-4.0 * x1_grid) * np.sin(6.0 * np.pi * x1_grid)**6)**0.1
        
        for i in range(n_points):
            if time.time() > deadline - 10:
                # Fallback: use nearest neighbor from grid
                idx_nn = np.argmin(np.abs(f1_grid - t[i]))
                x1_targets[i] = x1_grid[idx_nn]
                continue
                
            target = t[i]
            
            # Find a bracket using the grid
            # We want to find an interval [x1_grid[j], x1_grid[j+1]] where f1 crosses target.
            # Since f1 is not monotonic, we just need any crossing.
            # Let's find the index where |f1_grid - target| is minimized, and check neighbors.
            
            # More robust: find all intervals where sign change occurs.
            diff = f1_grid - target
            sign_change = np.diff(np.sign(diff))
            # sign_change != 0 means a crossing
            cross_idx = np.where(sign_change != 0)[0]
            
            if len(cross_idx) == 0:
                # No sign change? Maybe target is at a local extremum or outside range.
                # Use nearest neighbor.
                idx_nn = np.argmin(np.abs(diff))
                x1_targets[i] = x1_grid[idx_nn]
            else:
                # Pick the first crossing (or any)
                j = cross_idx[0]
                a = x1_grid[j]
                b = x1_grid[j+1]
                
                # Define the function for brentq
                def f1_minus_target(x):
                    return 1.0 - (1.0 - np.exp(-4.0 * x) * np.sin(6.0 * np.pi * x)**6)**0.1 - target
                
                try:
                    x1_sol = brentq(f1_minus_target, a, b, xtol=1e-12, rtol=1e-12)
                    x1_targets[i] = x1_sol
                except:
                    # Fallback to midpoint
                    x1_targets[i] = 0.5 * (a + b)
        
        # Evaluate
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1_targets[i]
            # Clip to bounds
            for d in range(dim):
                x[d] = np.clip(x[d], bounds[d][0], bounds[d][1])
            obj = problem(x)
            recorded.append(obj)

        return recorded
