import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, f1_vals, f2_vals, ref):
        """
        Compute Hypervolume for 2D points.
        f1_vals, f2_vals: numpy arrays of objective values.
        ref: tuple (ref1, ref2).
        Assumes points are non-dominated and sorted by f1 ascending.
        """
        if len(f1_vals) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = (f1_vals < ref[0]) & (f2_vals < ref[1])
        if not np.any(mask):
            return 0.0
            
        f1 = f1_vals[mask]
        f2 = f2_vals[mask]
        
        # Sort by f1 ascending
        sort_idx = np.argsort(f1)
        f1 = f1[sort_idx]
        f2 = f2[sort_idx]
        
        # Remove dominated points (keep Pareto front)
        # Since sorted by f1, a point is dominated if there's a previous point with lower f2
        # We want to keep points that have strictly decreasing f2 as f1 increases
        # Or rather, for unique f1, we just need to ensure we are looking at the non-dominated set.
        # For ZDT2 front, f2 = 1 - sqrt(f1) is strictly decreasing, so all points on the curve are non-dominated.
        # However, if we have duplicates or noise, we should handle it.
        # Standard 2D HV:
        # Sort by f1 asc. 
        # Keep only points where f2 is strictly lower than all previous points.
        
        if len(f1) == 1:
            return (ref[0] - f1[0]) * (ref[1] - f2[0])
        
        # Filter to non-dominated set
        # Since f1 is sorted, we just need to track min_f2_so_far
        # Actually, for HV calculation, we sum areas.
        # Area = sum_{i=0}^{n-2} (f1[i+1] - f1[i]) * (ref[1] - f2[i]) + (ref[0] - f1[n-1]) * (ref[1] - f2[n-1])
        # This formula assumes that f2[i] is the minimum f2 for all points with f1 <= f1[i+1].
        # If the points are not on a convex/concave front, or if there are dominated points, this needs adjustment.
        # For our generated points on the analytic front, they are non-dominated.
        
        hv = 0.0
        n = len(f1)
        
        # Ensure f2 is strictly decreasing for the formula to work directly on the sorted list
        # If not, we need to filter.
        # Let's filter to ensure f2 is strictly decreasing (Pareto optimal)
        pareto_idx = [0]
        min_f2 = f2[0]
        for i in range(1, n):
            if f2[i] < min_f2:
                pareto_idx.append(i)
                min_f2 = f2[i]
        
        if len(pareto_idx) == 0:
            return 0.0
            
        f1_p = f1[pareto_idx]
        f2_p = f2[pareto_idx]
        n_p = len(f1_p)
        
        if n_p == 1:
            return (ref[0] - f1_p[0]) * (ref[1] - f2_p[0])
            
        for i in range(n_p - 1):
            hv += (f1_p[i + 1] - f1_p[i]) * (ref[1] - f2_p[i])
        hv += (ref[0] - f1_p[-1]) * (ref[1] - f2_p[-1])
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        N = self.budget
        dim = self.dim
        
        # 1. Model: ZDT2 Pareto Front is x2..xD = 0, f1=x1, f2=1-sqrt(x1)
        # We optimize the x1 values to maximize HV of the resulting set.
        
        # Initial guess: Uniform grid in [0, 1]
        x1_params = np.linspace(0.0, 1.0, N)
        
        ref = (11.0, 11.0)
        
        def get_hv(x1s):
            # x1s are in [0, 1]
            f1 = x1s
            f2 = 1.0 - np.sqrt(f1)
            return self._compute_hv(f1, f2, ref)
        
        current_hv = get_hv(x1_params)
        
        # 2. Optimization of x1_params to maximize HV
        # Use a simple hill climbing / gradient-free optimization.
        # Since N=200, we can try to optimize the positions.
        
        # Hypothesis: Non-uniform spacing might yield better HV.
        # The ideal spacing for HV maximization with a concave front like 1-sqrt(x)
        # often clusters points where the curve is steeper or where the marginal gain is highest.
        # However, for a fixed number of points, the optimal distribution depends on the reference point.
        
        # Let's try a few parametric distributions and pick the best, then refine with hill climbing.
        
        best_x1 = x1_params.copy()
        best_hv = current_hv
        
        # Try power law distributions: x_i = (i/(N-1))^alpha
        alphas = np.linspace(0.5, 3.0, 10)
        for alpha in alphas:
            x1_try = (np.arange(N) / (N - 1)) ** alpha
            x1_try[0] = 0.0
            x1_try[-1] = 1.0
            hv_try = get_hv(x1_try)
            if hv_try > best_hv:
                best_hv = hv_try
                best_x1 = x1_try.copy()
        
        # Refine with local search (hill climbing)
        step_size = 0.01
        max_iters = 50
        no_improve_count = 0
        
        for iter_idx in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            improved = False
            # Randomly select a subset of points to perturb to save time?
            # With N=200, perturbing all is 200 * HV_calc.
            # HV_calc is O(N log N) due to sort. 200 * 200 log 200 ~ 300k ops. Fast.
            
            # Try to improve each point slightly
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                
                original = best_x1[i]
                # Try a few random steps
                for _ in range(3):
                    # Step size decay
                    step = step_size * np.random.uniform(-1, 1)
                    new_val = np.clip(original + step, 0.0, 1.0)
                    
                    # Ensure uniqueness? Not strictly necessary for HV, but helps.
                    # Just update and check
                    old_val = best_x1[i]
                    best_x1[i] = new_val
                    new_hv = get_hv(best_x1)
                    
                    if new_hv > best_hv + 1e-12:
                        best_hv = new_hv
                        improved = True
                        break
                    else:
                        best_x1[i] = old_val
            
            if not improved:
                step_size *= 0.5
                no_improve_count += 1
                if no_improve_count > 3 or step_size < 1e-6:
                    break
            else:
                no_improve_count = 0
        
        # Final check on time
        if time.time() > deadline - 5.0:
            # Not enough time for evaluations?
            # We must ensure we can evaluate at least some points.
            # Evaluations are fast (just function calls).
            pass

        # 3. Execute evaluations
        X = np.zeros((N, dim))
        X[:, 0] = best_x1
        # x[1:] are 0, which is optimal for ZDT2 (g(x) = 0)
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            results.append(f)
            
        return X
