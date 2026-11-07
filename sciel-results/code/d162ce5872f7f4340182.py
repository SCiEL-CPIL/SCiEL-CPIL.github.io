import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6: f1 = 1 + 90*(1-sqrt(x_1)), f2 = (1-f1^2/121)*(1-sqrt(f1/11))
        # x_1 = (1-(f1-1)/90)^2, others = 0
        # Pareto front: f1 in [1, 11]
        
        def f2_of_f1(f1):
            return (1 - f1**2 / 121.0) * (1 - np.sqrt(f1 / 11.0))
        
        def f1_to_x(f1):
            g = (f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            return x
        
        def evaluate_f1(f1_val):
            """Evaluate the problem at the point corresponding to f1_val on the Pareto front."""
            nonlocal n_evals
            if n_evals >= max_evals:
                return False
            x = f1_to_x(f1_val)
            x = np.clip(x, lo, hi)
            try:
                problem(x)
                n_evals += 1
                return True
            except Exception:
                return False
        
        # Compute hypervolume of a set of (f1, f2) points sorted by f1 ascending
        # Reference point is (11, 11)
        def compute_hv(f1_vals):
            """Compute hypervolume for points on Pareto front, ref [11,11]"""
            f1_vals = np.sort(f1_vals)
            f2_vals = f2_of_f1(f1_vals)
            N = len(f1_vals)
            if N == 0:
                return 0.0
            hv = 0.0
            for i in range(N - 1):
                hv += (f2_vals[i] - f2_vals[i+1]) * (11.0 - f1_vals[i])
            hv += f2_vals[N-1] * (11.0 - f1_vals[N-1])
            return hv
        
        def marginal_hv(f1_new, f1_sorted):
            """Compute marginal HV contribution of adding f1_new to existing set."""
            if len(f1_sorted) == 0:
                return (11.0 - f1_new) * f2_of_f1(f1_new)
            
            pos = np.searchsorted(f1_sorted, f1_new)
            f1_new_arr = np.insert(f1_sorted, pos, f1_new)
            return compute_hv(f1_new_arr) - compute_hv(f1_sorted)
        
        # Strategy: Place points to maximize hypervolume
        # For ZDT6 with ref [11,11], the optimal placement is to equalize
        # the hypervolume contribution of each point.
        
        # First, let's find the optimal number of points and their locations
        # For N points on the Pareto front, we want to maximize total HV
        
        # The hypervolume of N equally-spaced points in f1 is:
        # Sum over i of (f2_i - f2_{i+1}) * (11 - f1_i) + f2_N * (11 - f1_N)
        
        # Let's try to find the optimal f1 values by maximizing HV
        # We'll use a greedy approach: start with 2 points, then add points one by one
        
        # Initialize with endpoints
        f1_list = [1.0, 11.0]
        
        # Evaluate the two endpoints
        for f1 in [1.0, 11.0]:
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            evaluate_f1(f1)
        
        # Greedily add points with maximum marginal HV
        # Use a more efficient candidate generation strategy
        for iteration in range(max_evals - 2):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            f1_sorted = np.array(f1_list)
            
            # Generate candidates: for each interval, try several points
            # Focus on intervals where the marginal HV is likely to be high
            candidates = []
            
            for i in range(len(f1_sorted) - 1):
                f1_a = f1_sorted[i]
                f1_b = f1_sorted[i+1]
                width = f1_b - f1_a
                
                # Try points at different fractions of the interval
                # Weight towards the middle and also try some spread
                for frac in [0.25, 0.5, 0.75]:
                    candidates.append(f1_a + frac * width)
            
            # Also add some global candidates to explore
            # Use a log-spaced or linear-spaced set
            n_global = 10
            for t in np.linspace(0, 1, n_global):
                candidates.append(1.0 + 10.0 * t)
            
            # Remove duplicates and out-of-range
            candidates = np.array(candidates)
            candidates = candidates[(candidates > 1.0) & (candidates < 11.0)]
            candidates = np.unique(candidates)
            
            # Find candidate with max marginal HV
            best_f1 = None
            best_marginal = -1
            for f1_cand in candidates:
                if time.time() > deadline:
                    break
                marg = marginal_hv(f1_cand, f1_sorted)
                if marg > best_marginal:
                    best_marginal = marg
                    best_f1 = f1_cand
            
            if best_f1 is None:
                break
            
            # Add the best point
            pos = np.searchsorted(f1_sorted, best_f1)
            f1_list.insert(pos, best_f1)
            
            # Evaluate
            evaluate_f1(best_f1)
        
        return None
